//! What the interface says, and in which language it says it.
//!
//! Every string is a `&'static str` chosen by a `match`, so the whole table is compiled
//! into the binary with nothing to parse at startup and nothing to allocate while
//! drawing. A translation library would bring file loading, plural rules and gender
//! agreement to bear on forty labels of chess vocabulary.
//!
//! Everything here is plain ASCII, and a test enforces it. The font is three pixels wide
//! and five tall: an accent above a capital has nowhere to go, so French capitals drop
//! theirs the way they always have, German writes `ss` for the sharp s and spells out an
//! umlaut, and Portuguese does without its tilde. Every line also has a pixel budget --
//! sixteen characters in the in-game menu, twenty-five under the level row, twenty-eight
//! on the status band -- so these are not translations so much as the same thing said
//! again in a space that will not stretch. The layout tests are what hold that line.

/// The languages the interface speaks. `En` is the source: everything else is written
/// against it, and it is what an unrecognised system locale falls back to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    Fr,
    Es,
    De,
    It,
    Pt,
}

impl Lang {
    pub const ALL: [Lang; 6] = [Lang::En, Lang::Fr, Lang::Es, Lang::De, Lang::It, Lang::Pt];

    /// What the language calls itself, which is the only name a speaker of it can be
    /// counted on to recognise in a list.
    pub fn autonym(self) -> &'static str {
        match self {
            Lang::En => "english",
            Lang::Fr => "francais",
            Lang::Es => "espanol",
            Lang::De => "deutsch",
            Lang::It => "italiano",
            Lang::Pt => "portugues",
        }
    }

    /// The tag this language answers to.
    fn tag(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::Es => "es",
            Lang::De => "de",
            Lang::It => "it",
            Lang::Pt => "pt",
        }
    }

    /// Reads a BCP 47 tag, of which only the primary subtag is of any use here: `pt-BR`
    /// and `pt` get the same words, and a region this table cannot tell apart is a
    /// region it has nothing different to say to.
    fn from_tag(tag: &str) -> Option<Lang> {
        let primary = tag.trim().split(['-', '_']).next().unwrap_or("");
        Lang::ALL
            .into_iter()
            .find(|lang| primary.eq_ignore_ascii_case(lang.tag()))
    }
}

/// The language a tag asks for, or English where it names one this table cannot answer
/// in. For the headless captures, which are told a language on the command line.
#[cfg(not(target_arch = "wasm32"))]
pub fn for_tag(tag: &str) -> Lang {
    Lang::from_tag(tag).unwrap_or(Lang::En)
}

/// The language the machine is set to, or English where it says nothing this table
/// recognises.
#[cfg(all(feature = "gui", not(target_os = "haiku")))]
pub fn detect() -> Lang {
    sys_locale::get_locale()
        .and_then(|tag| Lang::from_tag(&tag))
        .unwrap_or(Lang::En)
}

/// Haiku's Locale preferences reach a POSIX process as the usual variables, which is
/// all sys-locale would read here anyway — asked directly instead of through a crate
/// that has never been built for the platform.
#[cfg(all(feature = "gui", target_os = "haiku"))]
pub fn detect() -> Lang {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .filter_map(std::env::var_os)
        .filter_map(|tag| Lang::from_tag(tag.to_str()?))
        .next()
        .unwrap_or(Lang::En)
}

/// The browser's preferences, in the order it ranks them, taken through the host: the
/// first one the interface can answer in wins. Reading them from Rust directly would
/// mean web-sys, and so wasm-bindgen, which this module is not brought up by.
#[cfg(all(feature = "gui-core", not(feature = "gui")))]
pub fn detect() -> Lang {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// Writes the languages the page asks for as a comma-separated list of BCP 47
        /// tags into `buf`, and returns its length, or -1 when there is nothing to say.
        fn gaia_locale(buf: *mut u8, cap: usize) -> i32;
    }

    // Longer than any real preference list needs, and truncation only costs the tail of
    // one, which is the part nobody would have been served by anyway.
    let mut buf = [0u8; 128];
    let written = unsafe { gaia_locale(buf.as_mut_ptr(), buf.len()) };
    if written <= 0 {
        return Lang::En;
    }
    let written = (written as usize).min(buf.len());
    // Tags are ASCII by definition, so anything else is a host that has misunderstood
    // the contract rather than a language to puzzle over.
    let Ok(list) = str::from_utf8(&buf[..written]) else {
        return Lang::En;
    };
    list.split(',').find_map(Lang::from_tag).unwrap_or(Lang::En)
}

/// Every phrase the interface shows, other than the level descriptions and the about
/// roll. The roll stays in English on purpose: it is a page of prose cut by hand to the
/// width of the panel, and the credits and the licence in it are canonical.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    // Title screen: the rows of the panel.
    White,
    Black,
    Level,
    Clock,
    Colours,
    Language,
    Sound,
    About,
    Play,
    // Title screen: what those rows are set to.
    Engine,
    Human,
    On,
    Off,
    Elo,
    // Title screen: the writing around the panel.
    Tagline,
    Since,
    MenuHint,
    // Clock choices that are words rather than numbers.
    ClockCustom,
    ClockNone,
    // Colour schemes.
    SchemeSlate,
    SchemeEmber,
    // The status band, which says what has just become true.
    Thinking,
    WhiteToMove,
    BlackToMove,
    WhiteWins,
    BlackWins,
    Stalemate,
    Draw,
    WhiteFlagged,
    BlackFlagged,
    WhiteCheck,
    BlackCheck,
    // The in-game menu.
    TakeBack,
    Restart,
    SoundOn,
    SoundOff,
    Quit,
    // The rest of the board screen.
    PromoteTo,
    GameHint,
    // Elsewhere.
    AboutHint,
    Loading,
}

/// One phrase, in one language.
///
/// The `match` is exhaustive on purpose: a key cannot be added without writing all six
/// columns, so no language can quietly fall back to English.
#[rustfmt::skip]
pub fn t(k: Key, l: Lang) -> &'static str {
    let row: [&'static str; 6] = match k {
        //                   english                    francais                       espanol                     deutsch                    italiano                        portugues
        Key::White        => ["white",                   "blancs",                      "blancas",                  "weiss",                   "bianco",                       "brancas"],
        Key::Black        => ["black",                   "noirs",                       "negras",                   "schwarz",                 "nero",                         "pretas"],
        Key::Level        => ["level",                   "niveau",                      "nivel",                    "stufe",                   "livello",                      "nivel"],
        Key::Clock        => ["clock",                   "cadence",                     "reloj",                    "bedenkzeit",              "tempo",                        "relogio"],
        Key::Colours      => ["colours",                 "couleurs",                    "colores",                  "farben",                  "colori",                       "cores"],
        Key::Language     => ["language",                "langue",                      "idioma",                   "sprache",                 "lingua",                       "idioma"],
        Key::Sound        => ["sound",                   "son",                         "sonido",                   "ton",                     "audio",                        "som"],
        Key::About        => ["about",                   "a propos",                    "acerca de",                "ueber",                   "info",                         "sobre"],
        Key::Play         => ["play",                    "jouer",                       "jugar",                    "spielen",                 "gioca",                        "jogar"],
        Key::Engine       => ["engine",                  "machine",                     "maquina",                  "computer",                "computer",                     "maquina"],
        Key::Human        => ["human",                   "humain",                      "humano",                   "mensch",                  "umano",                        "humano"],
        Key::On           => ["on",                      "oui",                         "si",                       "an",                      "si",                           "sim"],
        Key::Off          => ["off",                     "non",                         "no",                       "aus",                     "no",                           "nao"],
        Key::Elo          => ["elo",                     "elo",                         "elo",                      "elo",                     "elo",                          "elo"],
        Key::ClockCustom  => ["custom",                  "autre",                       "otro",                     "anderes",                 "altro",                        "outro"],
        Key::ClockNone    => ["none",                    "aucune",                      "ninguno",                  "keine",                   "nessuno",                      "nenhum"],
        Key::SchemeSlate  => ["slate",                   "ardoise",                     "pizarra",                  "schiefer",                "ardesia",                      "ardosia"],
        Key::SchemeEmber  => ["ember",                   "braise",                      "brasa",                    "glut",                    "brace",                        "brasa"],
        Key::Thinking     => ["thinking",                "reflexion",                   "pensando",                 "denkt nach",              "sto pensando",                 "pensando"],
        Key::WhiteToMove  => ["white to move",           "aux blancs de jouer",         "juegan blancas",           "weiss am zug",            "muove il bianco",              "brancas jogam"],
        Key::BlackToMove  => ["black to move",           "aux noirs de jouer",          "juegan negras",            "schwarz am zug",          "muove il nero",                "pretas jogam"],
        Key::Stalemate    => ["stalemate",               "pat",                         "ahogado",                  "patt",                    "stallo",                       "afogado"],
        Key::Draw         => ["draw",                    "partie nulle",                "tablas",                   "remis",                   "patta",                        "empate"],
        Key::WhiteFlagged => ["white flagged",           "blancs a court de temps",     "blancas sin tiempo",       "weiss hat keine zeit",    "bianco senza tempo",           "brancas sem tempo"],
        Key::BlackFlagged => ["black flagged",           "noirs a court de temps",      "negras sin tiempo",        "schwarz hat keine zeit",  "nero senza tempo",             "pretas sem tempo"],
        Key::WhiteCheck   => ["white to move - check",   "echec aux blancs",            "jaque a las blancas",      "schach - weiss am zug",   "scacco al bianco",             "xeque as brancas"],
        Key::BlackCheck   => ["black to move - check",   "echec aux noirs",             "jaque a las negras",       "schach - schwarz am zug", "scacco al nero",               "xeque as pretas"],
        Key::TakeBack     => ["take back",               "annuler",                     "deshacer",                 "zug zurueck",             "annulla",                      "voltar"],
        Key::Restart      => ["restart",                 "recommencer",                 "reiniciar",                "neu starten",             "ricomincia",                   "reiniciar"],
        Key::SoundOn      => ["sound on",                "son actif",                   "sonido si",                "ton an",                  "audio si",                     "som ligado"],
        Key::SoundOff     => ["sound off",               "son coupe",                   "sonido no",                "ton aus",                 "audio no",                     "som mudo"],
        Key::Quit         => ["quit",                    "quitter",                     "salir",                    "beenden",                 "esci",                         "sair"],
        Key::PromoteTo    => ["promote to",              "promouvoir en",               "promocionar a",            "umwandeln in",            "promuovi in",                  "promover a"],
        Key::GameHint     => ["esc menu  f flip",        "esc menu  f retourne",        "esc menu  f girar",        "esc menue  f drehen",     "esc menu  f ruota",            "esc menu  f girar"],
        Key::Loading      => ["loading the engine",      "chargement du moteur",        "cargando el motor",        "engine wird geladen",     "caricamento motore",           "carregando o motor"],
        Key::Since        => ["since 2003, on and off",  "depuis 2003, par a-coups",    "desde 2003, a ratos",      "seit 2003, mit pausen",   "dal 2003, a intermittenza",    "desde 2003, de vez em quando"],
        Key::WhiteWins    => ["checkmate - white wins",  "mat - les blancs gagnent",    "mate - ganan blancas",     "matt - weiss gewinnt",    "matto - vince il bianco",      "mate - brancas ganham"],
        Key::BlackWins    => ["checkmate - black wins",  "mat - les noirs gagnent",     "mate - ganan negras",      "matt - schwarz gewinnt",  "matto - vince il nero",        "mate - pretas ganham"],
        Key::AboutHint    => ["arrows scroll  esc closes",
                                                         "fleches defilent  esc ferme", "flechas mueven  esc sale", "pfeile scrollen  esc zu", "frecce scorrono  esc chiude",  "setas rolam  esc fecha"],
        Key::MenuHint     => ["arrows change  enter starts",
                                                         "fleches reglent  entree lance",
                                                                                        "flechas cambian  enter juega",
                                                                                                                    "pfeile aendern  enter startet",
                                                                                                                                               "frecce cambiano  invio avvia", "setas mudam  enter comeca"],
        Key::Tagline      => ["a chess engine with a board attached",
                                                         "un moteur d'echecs avec un echiquier",
                                                                                        "un motor de ajedrez con tablero",
                                                                                                                    "eine schachengine mit brett dran",
                                                                                                                                               "un motore di scacchi con scacchiera",
                                                                                                                                                                               "um motor de xadrez com tabuleiro"],
    };
    row[l as usize]
}

/// Who a rung plays like, said in the player's language.
///
/// English is not repeated here: [`crate::skill`] is where the ladder is described and
/// stays the one source for it -- the engine says the same thing over UCI, in the same
/// words, whatever the interface happens to be showing.
pub fn level_player(level: i32, l: Lang) -> &'static str {
    if l == Lang::En {
        return crate::skill::rating_for(level).player;
    }
    let i = level.clamp(1, LEVEL_PLAYERS.len() as i32) as usize - 1;
    debug_assert!(i < LEVEL_PLAYERS.len(), "level {level} is off the ladder");
    // One column short of the table above: English was answered before this point.
    LEVEL_PLAYERS[i][l as usize - 1]
}

/// The twenty rungs, in the five languages that are not the source. Each has to fit
/// under the level row, which leaves twenty-five characters.
#[rustfmt::skip]
const LEVEL_PLAYERS: [[&str; 5]; 20] = [
    //  francais                    espanol                    deutsch                    italiano                    portugues
    ["apprend les coups",       "acaba de aprender",       "kennt die zuege",         "ha imparato le mosse",     "aprendeu as jogadas"],
    ["decouvre les pieces",     "conoce las piezas",       "lernt die figuren",       "impara i pezzi",           "conhece as pecas"],
    ["ses premieres parties",   "primeras partidas",       "erste partien",           "prime partite",            "primeiras partidas"],
    ["debutant",                "principiante",            "anfaenger",               "principiante",             "iniciante"],
    ["debutant qui progresse",  "principiante avanzado",   "besserer anfaenger",      "principiante in crescita", "iniciante em evolucao"],
    ["joueur occasionnel",      "jugador ocasional",       "gelegenheitsspieler",     "giocatore occasionale",    "jogador ocasional"],
    ["amateur assidu",          "aficionado",              "eifriger amateur",        "amatore appassionato",     "amador dedicado"],
    ["joueur de club",          "jugador de club",         "vereinsspieler",          "giocatore di club",        "jogador de clube"],
    ["bon joueur de club",      "buen jugador de club",    "solider vereinsspieler",  "buon giocatore di club",   "bom jogador de clube"],
    ["joueur de club regulier", "jugador de club firme",   "fester vereinsspieler",   "giocatore di club solido", "jogador de clube firme"],
    ["joueur de club fort",     "jugador de club fuerte",  "starker vereinsspieler",  "forte giocatore di club",  "forte jogador de clube"],
    ["joueur de tournoi",       "jugador de torneo",       "turnierspieler",          "giocatore di torneo",      "jogador de torneio"],
    ["habitue des tournois",    "habitual de torneos",     "turniererfahren",         "veterano dei tornei",      "veterano de torneios"],
    ["candidat maitre",         "maestro candidato",       "kandidatenmeister",       "maestro candidato",        "mestre candidato"],
    ["expert",                  "experto",                 "experte",                 "esperto",                  "especialista"],
    ["maitre national",         "maestro nacional",        "nationaler meister",      "maestro nazionale",        "mestre nacional"],
    ["maitre confirme",         "maestro fuerte",          "starker meister",         "maestro forte",            "mestre forte"],
    ["maitre international",    "maestro internacional",   "internationaler meister", "maestro internazionale",   "mestre internacional"],
    ["grand maitre",            "gran maestro",            "grossmeister",            "grande maestro",           "grande mestre"],
    ["pleine puissance",        "fuerza maxima",           "volle staerke",           "forza piena",              "forca total"],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::font;
    use crate::skill::FULL_STRENGTH;

    /// Listed rather than derived: there is no way to walk an enum, and a key left out
    /// of this list is a key no test ever measures.
    #[rustfmt::skip]
    pub const KEYS: [Key; 41] = [
        Key::White, Key::Black, Key::Level, Key::Clock, Key::Colours, Key::Language,
        Key::Sound, Key::About, Key::Play, Key::Engine, Key::Human, Key::On, Key::Off,
        Key::Elo, Key::Tagline, Key::Since, Key::MenuHint, Key::ClockCustom,
        Key::ClockNone, Key::SchemeSlate, Key::SchemeEmber, Key::Thinking,
        Key::WhiteToMove, Key::BlackToMove, Key::WhiteWins, Key::BlackWins,
        Key::Stalemate, Key::Draw, Key::WhiteFlagged, Key::BlackFlagged, Key::WhiteCheck,
        Key::BlackCheck, Key::TakeBack, Key::Restart, Key::SoundOn, Key::SoundOff,
        Key::Quit, Key::PromoteTo, Key::GameHint, Key::AboutHint, Key::Loading,
    ];

    /// Every phrase, in every language, including the level descriptions.
    fn everything() -> Vec<(Lang, &'static str)> {
        let mut out = Vec::new();
        for lang in Lang::ALL {
            for key in KEYS {
                out.push((lang, t(key, lang)));
            }
            out.push((lang, lang.autonym()));
            for level in 1..=FULL_STRENGTH {
                out.push((lang, level_player(level, lang)));
            }
        }
        out
    }

    /// The font has no accented forms and cannot be given any: a mark above a capital
    /// has nowhere to go in five rows. A character it has never heard of prints as a
    /// blank, which is a hole in the middle of a word that nothing at runtime reports.
    #[test]
    fn every_phrase_can_be_written_in_this_font() {
        for (lang, text) in everything() {
            assert!(!text.is_empty(), "{lang:?} says nothing somewhere");
            for c in text.chars() {
                assert!(c.is_ascii(), "{lang:?}: {c:?} is not ascii, in {text:?}");
                assert!(font::has_glyph(c), "{lang:?}: no glyph for {c:?} in {text:?}");
            }
        }
    }

    /// Nothing may be left in English by accident. Words that are the same in two
    /// languages are real -- `elo` is one, and `reiniciar` is Spanish and Portuguese
    /// both -- so this weighs the whole set rather than each phrase on its own.
    #[test]
    fn every_language_is_actually_a_language() {
        for lang in Lang::ALL {
            if lang == Lang::En {
                continue;
            }
            let english = KEYS
                .into_iter()
                .filter(|key| t(*key, lang) == t(*key, Lang::En))
                .count();
            assert!(
                english < KEYS.len() / 4,
                "{lang:?} still says {english} of {} phrases in english",
                KEYS.len()
            );
        }
    }

    #[test]
    fn a_system_tag_finds_its_language() {
        assert_eq!(Lang::from_tag("fr-FR"), Some(Lang::Fr));
        assert_eq!(Lang::from_tag("fr"), Some(Lang::Fr));
        // Windows hands out underscores, and case is not part of a tag's meaning.
        assert_eq!(Lang::from_tag("de_DE"), Some(Lang::De));
        assert_eq!(Lang::from_tag("PT-br"), Some(Lang::Pt));
        // A region with nothing of its own to say still gets the language.
        assert_eq!(Lang::from_tag("es-419"), Some(Lang::Es));
        // Anything the table cannot answer in is left for the caller to fall back on.
        assert_eq!(Lang::from_tag("ru-RU"), None);
        assert_eq!(Lang::from_tag(""), None);
    }

    #[test]
    fn the_ladder_is_described_in_every_language() {
        assert_eq!(LEVEL_PLAYERS.len(), FULL_STRENGTH as usize, "the ladder changed length");
        for lang in Lang::ALL {
            // Off either end of the ladder is a clamp rather than a panic: the level row
            // is walked by the player, and the search clamps the same way.
            assert_eq!(level_player(0, lang), level_player(1, lang));
            assert_eq!(level_player(99, lang), level_player(FULL_STRENGTH, lang));
        }
        for level in 1..=FULL_STRENGTH {
            assert_eq!(
                level_player(level, Lang::En),
                crate::skill::rating_for(level).player,
                "english drifted from the ladder at level {level}"
            );
        }
    }
}
