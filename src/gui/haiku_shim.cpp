// The native Haiku host for the interface: a window that blits the CPU framebuffer,
// an input snapshot for the Rust loop to poll, and a small mixer behind the two audio
// entry points the browser host also implements. Compiled by build.rs for Haiku GUI
// builds only; the Rust side of this ABI is src/gui/haiku.rs, and the key bit indices
// below mirror its `key` module — change one and the other must follow.
//
// Threading: BApplication runs its message loop on a spawned thread (each BWindow has
// a looper thread of its own regardless), leaving the process's main thread to the
// Rust frame loop, which owns the pace. Everything shared crosses under one mutex or
// as an atomic; the audio mixer callback runs on the media kit's thread and takes the
// voice lock only.

#include <Application.h>
#include <Bitmap.h>
#include <InterfaceDefs.h>
#include <Screen.h>
#include <Message.h>
#include <OS.h>
#include <Rect.h>
#include <SoundPlayer.h>
#include <View.h>
#include <Window.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// --- Shared state -----------------------------------------------------------------

// Mirror of src/gui/haiku.rs `Snapshot`, #[repr(C)].
struct GaiaSnapshot {
    float mouse_x;
    float mouse_y;
    uint32_t keys_down;
    uint32_t keys_pressed;
    uint8_t buttons;
    uint8_t pressed;
    uint8_t inside;
    uint8_t pad;
};

namespace {

// Key bit indices, mirroring haiku.rs `key`.
enum GaiaKey : uint32_t {
    GK_LEFT = 0, GK_RIGHT, GK_UP, GK_DOWN,
    GK_A, GK_D, GK_W, GK_S,
    GK_X, GK_V, GK_ENTER, GK_SPACE,
    GK_Z, GK_C, GK_ESCAPE, GK_BACKSPACE,
    GK_F, GK_TAB, GK_R, GK_U, GK_N, GK_M,
    GK_DIGIT_1, // eight consecutive bits, digits 1 through 8
    GK_COUNT = GK_DIGIT_1 + 8,
};

// Haiku raw key codes name physical positions, which is exactly what the interface
// binds (see input.rs: the letters are positions, enter and escape the advertised
// pair). Values from the keyboard appendix of the Haiku Book.
int32_t key_bit(int32_t raw) {
    switch (raw) {
        case 0x61: return GK_LEFT;
        case 0x63: return GK_RIGHT;
        case 0x57: return GK_UP;
        case 0x62: return GK_DOWN;
        case 0x3c: return GK_A;
        case 0x3e: return GK_D;
        case 0x28: return GK_W;
        case 0x3d: return GK_S;
        case 0x4d: return GK_X;
        case 0x4f: return GK_V;
        case 0x47: return GK_ENTER;
        case 0x5e: return GK_SPACE;
        case 0x4c: return GK_Z;
        case 0x4e: return GK_C;
        case 0x01: return GK_ESCAPE;
        case 0x1e: return GK_BACKSPACE;
        case 0x3f: return GK_F;
        case 0x26: return GK_TAB;
        case 0x2a: return GK_R;
        case 0x2d: return GK_U;
        case 0x51: return GK_N;
        case 0x52: return GK_M;
        default:
            if (raw >= 0x12 && raw <= 0x19) // number row, 1 through 8
                return GK_DIGIT_1 + (raw - 0x12);
            return -1;
    }
}

std::mutex g_input_lock;
GaiaSnapshot g_input = {};

std::atomic<float> g_view_w{0.0f};
std::atomic<float> g_view_h{0.0f};
std::atomic<bool> g_quit{false};
std::atomic<bool> g_cursor_hidden{false};

class GaiaWindow;
GaiaWindow* g_window = nullptr;
thread_id g_app_thread = -1;

// The canvas size, learnt from the first frame handed over.
int32_t g_fb_w = 0;
int32_t g_fb_h = 0;

// --- Window and view --------------------------------------------------------------

// Where the canvas sits in the view: the same arithmetic as input.rs `viewport`, so
// what is drawn under a pixel is what a click on that pixel is taken to mean.
static void viewport(float vw, float vh, float* ox, float* oy, float* scale) {
    if (g_fb_w <= 0 || g_fb_h <= 0) {
        *ox = *oy = 0.0f;
        *scale = 1.0f;
        return;
    }
    float s = std::fmax(std::fmin(vw / g_fb_w, vh / g_fb_h), 0.01f);
    bool fits = std::fabs(vw / g_fb_w - vh / g_fb_h) < 0.01f;
    if (!fits && s >= 2.0f)
        s = std::floor(s);
    *ox = std::floor((vw - g_fb_w * s) * 0.5f);
    *oy = std::floor((vh - g_fb_h * s) * 0.5f);
    *scale = s;
}

class GaiaView : public BView {
public:
    GaiaView(BRect frame)
        : BView(frame, "canvas", B_FOLLOW_ALL_SIDES, B_WILL_DRAW),
          fBitmap(nullptr) {
        // app_server clears with the view colour before Draw runs, so black here is
        // what keeps a resize from flashing white around the picture.
        SetViewColor(0, 0, 0);
    }

    ~GaiaView() override { delete fBitmap; }

    // Copies one RGBA frame in, converting to the B_RGB32 byte order (BGRX).
    // Caller holds the window looper lock.
    void SetFrame(const uint8_t* rgba, int32_t w, int32_t h) {
        if (fBitmap == nullptr || fBitmap->Bounds().IntegerWidth() + 1 != w
            || fBitmap->Bounds().IntegerHeight() + 1 != h) {
            delete fBitmap;
            fBitmap = new BBitmap(BRect(0, 0, w - 1, h - 1), B_RGB32);
        }
        uint8_t* bits = static_cast<uint8_t*>(fBitmap->Bits());
        int32_t row = fBitmap->BytesPerRow();
        for (int32_t y = 0; y < h; y++) {
            const uint8_t* src = rgba + y * w * 4;
            uint8_t* dst = bits + y * row;
            for (int32_t x = 0; x < w; x++) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = 255;
                src += 4;
                dst += 4;
            }
        }
        Invalidate();
    }

    void Draw(BRect) override {
        if (fBitmap == nullptr)
            return;
        BRect b = Bounds();
        float ox, oy, scale;
        viewport(b.Width() + 1, b.Height() + 1, &ox, &oy, &scale);
        // DrawBitmap scales by point sampling unless bilinear filtering is asked
        // for, and point sampling is the only enlargement that leaves pixel art
        // alone — so the default is exactly right.
        BRect dst(ox, oy, ox + g_fb_w * scale - 1, oy + g_fb_h * scale - 1);
        DrawBitmap(fBitmap, fBitmap->Bounds(), dst);
    }

    void MouseMoved(BPoint where, uint32 transit, const BMessage*) override {
        std::lock_guard<std::mutex> lock(g_input_lock);
        g_input.mouse_x = where.x;
        g_input.mouse_y = where.y;
        g_input.inside = (transit == B_INSIDE_VIEW || transit == B_ENTERED_VIEW) ? 1 : 0;
    }

    void MouseDown(BPoint where) override {
        int32 buttons = 0;
        if (Window()->CurrentMessage() != nullptr)
            Window()->CurrentMessage()->FindInt32("buttons", &buttons);
        if ((buttons & B_PRIMARY_MOUSE_BUTTON) == 0)
            return;
        // Keep the pointer's events coming while a piece is carried outside the
        // window; letting go out there must still be seen as letting go.
        SetMouseEventMask(B_POINTER_EVENTS, 0);
        std::lock_guard<std::mutex> lock(g_input_lock);
        g_input.mouse_x = where.x;
        g_input.mouse_y = where.y;
        g_input.buttons |= 1;
        g_input.pressed = 1;
    }

    void MouseUp(BPoint where) override {
        std::lock_guard<std::mutex> lock(g_input_lock);
        g_input.mouse_x = where.x;
        g_input.mouse_y = where.y;
        g_input.buttons &= ~uint8_t(1);
    }

private:
    BBitmap* fBitmap;
};

class GaiaWindow : public BWindow {
public:
    GaiaWindow(const char* title, float w, float h)
        : BWindow(BRect(0, 0, w - 1, h - 1), title, B_TITLED_WINDOW,
                  B_ASYNCHRONOUS_CONTROLS | B_QUIT_ON_WINDOW_CLOSE) {
        fView = new GaiaView(Bounds());
        AddChild(fView);
        fView->MakeFocus(true);
        g_view_w.store(w);
        g_view_h.store(h);
        CenterOnScreen();
    }

    // The close button asks; the Rust loop answers by breaking and calling
    // gaia_shim_quit, so the window is never torn down under a frame in flight.
    bool QuitRequested() override {
        g_quit.store(true);
        return false;
    }

    // Keys are read off the window rather than the view so focus can never wander
    // away from the board. Raw codes only: positions, not letters.
    // GAIA_SHIM_LOG_KEYS=1 in the environment prints every raw code to stderr,
    // which is how the mapping table gets verified against a live keyboard.
    void DispatchMessage(BMessage* msg, BHandler* target) override {
        switch (msg->what) {
            case B_KEY_DOWN:
            case B_UNMAPPED_KEY_DOWN: {
                int32 raw = 0;
                if (msg->FindInt32("key", &raw) == B_OK) {
                    static const bool log = getenv("GAIA_SHIM_LOG_KEYS") != nullptr;
                    if (log)
                        fprintf(stderr, "key down raw=0x%02x bit=%d\n", (unsigned)raw,
                                (int)key_bit(raw));
                    int32_t bit = key_bit(raw);
                    if (bit >= 0) {
                        std::lock_guard<std::mutex> lock(g_input_lock);
                        uint32_t mask = uint32_t(1) << bit;
                        // Auto-repeats arrive as more key-downs; a press edge is
                        // only the first of them.
                        if ((g_input.keys_down & mask) == 0)
                            g_input.keys_pressed |= mask;
                        g_input.keys_down |= mask;
                    }
                }
                break;
            }
            case B_KEY_UP:
            case B_UNMAPPED_KEY_UP: {
                int32 raw = 0;
                if (msg->FindInt32("key", &raw) == B_OK) {
                    static const bool log = getenv("GAIA_SHIM_LOG_KEYS") != nullptr;
                    if (log)
                        fprintf(stderr, "key up   raw=0x%02x bit=%d\n", (unsigned)raw,
                                (int)key_bit(raw));
                    int32_t bit = key_bit(raw);
                    if (bit >= 0) {
                        std::lock_guard<std::mutex> lock(g_input_lock);
                        g_input.keys_down &= ~(uint32_t(1) << bit);
                    }
                }
                break;
            }
        }
        BWindow::DispatchMessage(msg, target);
    }

    // Snaps the window back to the canvas's shape, keeping whichever side was
    // dragged. Only a size that actually changed is corrected, so a window manager
    // that refuses is asked once and then left alone.
    void FrameResized(float w, float h) override {
        g_view_w.store(w + 1);
        g_view_h.store(h + 1);
        if (g_fb_w <= 0 || g_fb_h <= 0)
            return;
        float aspect = float(g_fb_w) / float(g_fb_h);
        float sw = w + 1, sh = h + 1;
        float moved_w = std::fabs(sw - fLast.x), moved_h = std::fabs(sh - fLast.y);
        fLast = BPoint(sw, sh);
        if (moved_w < 0.5f && moved_h < 0.5f)
            return;
        float want_w, want_h;
        if (moved_w >= moved_h) {
            want_w = sw;
            want_h = std::round(sw / aspect);
        } else {
            want_w = std::round(sh * aspect);
            want_h = sh;
        }
        if (std::fabs(want_w - sw) >= 1.0f || std::fabs(want_h - sh) >= 1.0f)
            ResizeTo(want_w - 1, want_h - 1);
        BWindow::FrameResized(w, h);
    }

    GaiaView* fView;

private:
    BPoint fLast{0, 0};
};

// --- Application ------------------------------------------------------------------

int32 app_thread(void*) {
    be_app->Lock();
    be_app->Run();
    return 0;
}

// Creates the BApplication once, reporting whether app_server answered. Called by
// the display probe as well as by init: an engine started over SSH must learn "no
// window" here, not by aborting later.
bool ensure_app() {
    static bool tried = false;
    static bool ok = false;
    if (tried)
        return ok;
    tried = true;
    status_t error = B_OK;
    new BApplication("application/x-vnd.GaiaChess", &error);
    ok = (error == B_OK);
    return ok;
}

} // namespace

// --- C ABI ------------------------------------------------------------------------

extern "C" {

int gaia_shim_display_available(void) {
    return ensure_app() ? 1 : 0;
}

int gaia_shim_init(const char* title, int fb_w, int fb_h, int scale) {
    if (!ensure_app())
        return 0;
    be_app->Unlock(); // constructed locked; Run() on its own thread relocks
    g_app_thread = spawn_thread(app_thread, "gaia app loop", B_NORMAL_PRIORITY, nullptr);
    if (g_app_thread < 0)
        return 0;
    resume_thread(g_app_thread);

    // The asked-for scale, or the largest whole multiple that still fits the
    // screen with a little air for the tab and the Deskbar: a window taller
    // than the desktop can never be dragged back into shape.
    BRect screen = BScreen().Frame();
    while (scale > 1
           && (fb_w * scale > screen.Width() - 16 || fb_h * scale > screen.Height() - 48))
        scale--;

    g_window = new GaiaWindow(title, float(fb_w * scale), float(fb_h * scale));
    g_window->Show();
    return 1;
}

void gaia_shim_frame(const uint8_t* rgba, int w, int h) {
    g_fb_w = w;
    g_fb_h = h;
    if (g_window != nullptr && g_window->Lock()) {
        g_window->fView->SetFrame(rgba, w, h);
        g_window->Unlock();
    }
}

void gaia_shim_input(GaiaSnapshot* out) {
    std::lock_guard<std::mutex> lock(g_input_lock);
    *out = g_input;
    // The edges are the poll's to keep; the level state stays.
    g_input.keys_pressed = 0;
    g_input.pressed = 0;
}

void gaia_shim_view_size(float* w, float* h) {
    *w = g_view_w.load();
    *h = g_view_h.load();
}

void gaia_shim_show_cursor(int show) {
    // Balanced by hand: HideCursor stacks inside app_server, and the interface asks
    // every frame.
    bool hidden = g_cursor_hidden.load();
    if (show != 0 && hidden) {
        be_app->ShowCursor();
        g_cursor_hidden.store(false);
    } else if (show == 0 && !hidden) {
        be_app->HideCursor();
        g_cursor_hidden.store(true);
    }
}

int gaia_shim_should_quit(void) {
    return g_quit.load() ? 1 : 0;
}

// --- Audio ------------------------------------------------------------------------

namespace {

struct Voice {
    int32_t clip = -1;
    size_t pos = 0;
    float gain = 0.0f;
};

std::mutex g_audio_lock;
std::vector<std::vector<float>> g_clips;
Voice g_voices[16];
BSoundPlayer* g_player = nullptr;
uint32_t g_rate = 0;
bool g_audio_started = false;
thread_id g_audio_thread = -1;

// Mixes the live voices into the card's buffer. Media kit thread.
void mix(void*, void* buffer, size_t size, const media_raw_audio_format& format) {
    float* out = static_cast<float*>(buffer);
    size_t frames = size / (sizeof(float) * format.channel_count);
    std::memset(buffer, 0, size);
    std::lock_guard<std::mutex> lock(g_audio_lock);
    for (Voice& v : g_voices) {
        if (v.clip < 0)
            continue;
        const std::vector<float>& clip = g_clips[v.clip];
        for (size_t i = 0; i < frames && v.pos < clip.size(); i++, v.pos++) {
            float s = clip[v.pos] * v.gain;
            for (uint32 c = 0; c < format.channel_count; c++)
                out[i * format.channel_count + c] += s;
        }
        if (v.pos >= clip.size())
            v.clip = -1;
    }
}

// Opens the device off the critical path: talking to the media server can hang for
// tens of seconds — or take the server down — on a machine with no sound hardware,
// and the board must not be held hostage by that. Requests queued meanwhile start
// playing the moment the player lands; on a soundless machine they simply never do.
int32 audio_open_thread(void*) {
    media_raw_audio_format format = media_raw_audio_format::wildcard;
    format.frame_rate = float(g_rate);
    format.channel_count = 1;
    format.format = media_raw_audio_format::B_AUDIO_FLOAT;
    format.byte_order = B_MEDIA_HOST_ENDIAN;
    format.buffer_size = 1024 * sizeof(float);
    BSoundPlayer* player = new BSoundPlayer(&format, "gaiachess", mix);
    if (player->InitCheck() != B_OK) {
        // A machine with no sound should still be able to play chess.
        delete player;
        return 0;
    }
    player->Start();
    player->SetHasData(true);
    std::lock_guard<std::mutex> lock(g_audio_lock);
    g_player = player;
    return 0;
}

} // namespace

void gaia_sfx_register(uint32_t id, const float* samples, size_t len, uint32_t rate) {
    bool start = false;
    {
        std::lock_guard<std::mutex> lock(g_audio_lock);
        if (g_clips.size() <= id)
            g_clips.resize(id + 1);
        // Copied during the call, as the Rust side is promised.
        g_clips[id].assign(samples, samples + len);
        g_rate = rate;
        if (!g_audio_started) {
            g_audio_started = true;
            start = true;
        }
    }
    if (start) {
        g_audio_thread =
            spawn_thread(audio_open_thread, "gaia audio open", B_NORMAL_PRIORITY, nullptr);
        if (g_audio_thread >= 0)
            resume_thread(g_audio_thread);
    }
}

void gaia_sfx_play(uint32_t id, float gain) {
    std::lock_guard<std::mutex> lock(g_audio_lock);
    if (id >= g_clips.size() || g_clips[id].empty())
        return;
    for (Voice& v : g_voices) {
        if (v.clip < 0) {
            v.clip = int32_t(id);
            v.pos = 0;
            v.gain = gain;
            return;
        }
    }
    // Every voice busy: the newest request is the one the player is acting on now,
    // so the oldest is the one to give up.
    Voice* oldest = &g_voices[0];
    for (Voice& v : g_voices)
        if (v.pos > oldest->pos)
            oldest = &v;
    oldest->clip = int32_t(id);
    oldest->pos = 0;
    oldest->gain = gain;
}

void gaia_shim_quit(void) {
    // The opener may still be talking to the media server; let it land or fail
    // before the player it might publish is torn down.
    if (g_audio_thread >= 0) {
        status_t ret;
        wait_for_thread(g_audio_thread, &ret);
    }
    if (g_player != nullptr) {
        g_player->Stop();
        delete g_player;
        g_player = nullptr;
    }
    if (g_window != nullptr && g_window->Lock()) {
        g_window->Quit(); // deletes the window
        g_window = nullptr;
    }
    if (be_app != nullptr) {
        be_app->PostMessage(B_QUIT_REQUESTED);
        if (g_app_thread >= 0) {
            status_t ret;
            wait_for_thread(g_app_thread, &ret);
        }
    }
}

} // extern "C"
