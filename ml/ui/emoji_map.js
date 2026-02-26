/**
 * UI emoji/symbol mapper.
 *
 * Keep this file ASCII-only by using Unicode escape sequences so accidental
 * file re-encoding cannot corrupt displayed glyphs.
 */
(function attachUiGlyphs(global) {
  const emoji = Object.freeze({
    cards: "\ud83c\udfb4",
    scroll: "\ud83d\udccb",
    play: "\ud83c\udccf",
    money: "\ud83d\udcb0",
    blocked: "\u26d4",
    handshake: "\ud83e\udd1d",
    trophy: "\ud83c\udfc6",
    question: "\u2753",
    check: "\u2705",
    cross: "\u274c",
    trick_win: "\ud83c\udfc5",
    reload: "\u21bb",
    play_button: "\u25b6"
  });

  const symbols = Object.freeze({
    em_dash: "\u2014",
    right_arrow: "\u2192",
    middle_dot: "\u00b7",
    minus_sign: "\u2212"
  });

  global.UI_GLYPHS = Object.freeze({ emoji, symbols });
})(window);
