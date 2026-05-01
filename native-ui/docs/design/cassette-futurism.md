# Cassette futurism — design language

Unified visual and interaction language for the Notebook native UI. Tradeoffs are judged against **clarity**, **warmth**, **ease of use**, **tactile** presence, and **inspiration**; novelty must not sacrifice the first three. Tactile strengthens *ease of use* and *warmth* when it reinforces real affordances—it must not replace clarity (no mystery meat interaction).

---

## Part 1 — Motivation for the design role

### Why a unified language

Notebook is a tool people use for long sessions: reading, writing, navigating graphs, and chatting with a model. The interface should read as **one coherent instrument**—a console understood at a glance—not an arbitrary stack of widgets, and **not a faceless chat surface**. A named design role gives a shared vocabulary: when in doubt, ask whether a change improves clarity, warmth, ease of use, tactile presence, or inspiration, and reject changes that sacrifice the first three for novelty.

### Clarity

The UI must make structure obvious: chrome vs content, active vs idle, where one thought ends and another begins. Clarity is not minimalism for its own sake; dense workspaces are acceptable when **edges, rhythm, and hierarchy** read instantly. Prefer strong figure–ground separation, predictable placement, and typography that signals role (title vs body vs metadata) without relying on color alone.

### Warmth

A research and writing surface should not feel like a hostile machine room. “Cassette futurism” here means **human-scale optimism**: the future as imagined from institutional, slightly worn materials—paper, plastic, phosphor—not sterile neon maximalism. Warmth comes from **muted earth and cream neutrals**, **soft containment** (panels and bezels), and **restraint** in motion and glow so the app feels steady and trustworthy.

### Ease of use

Ease is clarity and warmth plus **forgiveness**: readable type, comfortable contrast, easy hit targets, states that are easy to parse. Long reading and writing need **low-glare backgrounds**, **disciplined accents**, and **no thin colored lines as the only signal**. Chunky controls echo hardware; generous margins echo manuals and posters that leave room for the eye to rest.

### Inspiration (cassette futurism)

The emotional reference is **late-century institutional tech**: mission control, lab instruments, travel posters, CRT readouts, cassette shells—**clean geometry**, **thick rules**, occasional **bands suggesting horizon or depth**, and **small controls on large quiet fields**. Inspiration is atmosphere, not pastiche: borrow **structure and materials**, not effects that hurt readability.

### Tactile

The product should feel like **using a tangible machine**: switches, plates, and readouts you can *locate* and *trust*, not a monolithic “abstract chatbot” with flat, stateless panels. Tangibility comes from **motion that implies mass** (short, damped, purposeful—see [Motion](#motion--tape-deck--mechanical-feel)), **visible state** (pressed vs idle buttons, selected rows, graph structure and selection, where focus lives), **interaction affordances** (drag previews / placement ghosts, resize handles, obvious hit targets), and **light skeuomorphism** where it helps: **layered depth** (stacked chrome vs content), **soft drop shadows** or inset highlights on floating panels and modals, and **thick bezels** so regions read as physical parts. Tactile effects stay **subtle**—enough to cue the hand and eye, never enough to drown readouts or emulate tacky 3D chrome everywhere.

---

## Part 2 — Map: goals to specific design principles

| Goal | Principle | What we do in practice |
|------|-----------|------------------------|
| **Clarity** | Poster margins and gutters | Wider outer margins and consistent gutters so regions read as plates on a desk, not edge-to-edge noise. |
| **Clarity** | Modular mission-control layout | Break complex screens into gridded modules with clear titles and separators; density inside a module, calm at boundaries. |
| **Clarity** | Thick decisive structure | Use instrument-grade strokes and nested frames (outer shell → well → control) so hierarchy is seen before it is read. |
| **Clarity** | Readout grammar | Chrome (labels, frames, section titles) vs readouts (body text, transcripts): distinct size/weight steps; color is secondary to structure. |
| **Clarity** | Horizon as vertical rhythm | Stack as **sky (chrome) → instrument band → main viewport → deck (composer)** so orientation matches the horizon motif without cluttering content. |
| **Warmth** | Earth-and-cream neutrals | Ground the UI in muted paper, beige, charcoal, cool slate shadows; avoid pure white and hyper-saturated large fills. |
| **Warmth** | Matte, tactile surfaces | Prefer matte fills, soft vignette, subtle grain over glassy blur; suggest plastic and paper. |
| **Warmth** | Chunky, molded geometry | Rounded rectangles and circles from a consistent radius family; controls feel molded. |
| **Warmth** | Disciplined accents | One warm and one cool accent for emphasis and state; accents are impulses, not wallpaper. |
| **Ease of use** | Earth tones against text bleed | Large text areas on low-chroma or deep bulkhead backgrounds; cream/light text on deep panels for primary reading. |
| **Ease of use** | Comfortable contrast | Strong text vs panel contrast; phosphor or scanline flavor stays low amplitude and behind content. |
| **Ease of use** | Physical hit targets | Buttons and handles sized like hardware; target slightly larger than the icon; thick outlines mark affordances. |
| **Ease of use** | Redundant state cues | Selection and focus use outline, weight, or position as well as color. |
| **Ease of use** | Mechanical motion | Short, purposeful transitions; damped easing—tape capstan, not bounce. |
| **Inspiration** | Horizon and speed lines (structural) | Parallel or converging bands as header/footer trim, deck edges, or parallax—not behind body copy. |
| **Inspiration** | Large-field motifs | Quiet backgrounds and slow depth so controls feel small at a big console. |
| **Inspiration** | Simple geometric marks | Flat geometric glyphs or empty states that never fight text. |
| **Inspiration** | CRT discipline | Single atmospheric layer for glow/grain; readability wins. |
| **Tactile** | Mechanical motion as mass | Subtle animation (springs, scroll inertia) so controls feel weighted; prefer `AnimationPreset::Mechanical`-class motion on chrome. |
| **Tactile** | Explicit interaction states | Pressed / hover / active / disabled always distinguishable; graph and lists show **persistent structure** (edges, selection), not only transient AI output. |
| **Tactile** | Drag and placement cues | Drags show **ghosts or previews** where the object will land; don’t rely on silent snap alone. |
| **Tactile** | Depth without clutter | **Drop shadow** or soft elevation on modals, popovers, and floating toolbars; **inset wells** for embedded controls; avoid universal faux-3D. |
| **Tactile** | Chunky affordances | Thick rims, obvious handles, and redundant cues (outline + motion) so the tool feels operable by hand. |

### Token roles (implementation map)

These names tie rhetoric to modules in [`src/ui/style.rs`](../../src/ui/style.rs):

| Role | Meaning | Primary `style` modules / tokens |
|------|---------|----------------------------------|
| **Horizon** | Chrome above content: header band, tab folder, instrument rule under sky | `stroke::INSTRUMENT_RULE_PX`, `chrome::TAB_*`, `border::INSTRUMENT_RULE` |
| **Bulkhead** | Deep panel shells and wells (viewer chrome, popups) | `bg::PRIMARY`, `bg::CHROME_DECK`, `bg::PANEL_WELL`, `backdrop::*` |
| **Readout** | Body and labels on panels | `text::*`, `font_size::*`, `bg::INPUT` / `INPUT_FOCUSED` |
| **Impulse** | Primary user/system emphasis (warm pop, cool accent) | `accent::POP`, `accent::POP_COOL`, `accent::PHOSPHOR` |
| **Warning** | Destructive or alert emphasis | `accent::WARNING`, `button::DANGER_*` |
| **Elevation** | Directional lighting — drop shadows, inset shadows, specular rims and surface sheens — all driven by one key light at `-45°` (upper-left) | Offsets toward `+135°` by [`style::elevation::SHADOW_SIZE_PX`](../../src/ui/style.rs). Four primitives in [`ui/shadow.rs`](../../src/ui/shadow.rs): `ShadowSpec` / `InnerShadowSpec` / `BorderHighlightSpec` / `SurfaceHighlightSpec`. Each is queued as one quad via the matching `Renderer::queue_*` method and evaluated O(1) per pixel by its own sentinel branch (`bubble == 3..=6`) in [`ui_shader.wgsl`](../../src/gfx/shaders/ui_shader.wgsl): `erf7`-SDF for shadows, SDF-gradient Lambert for specular highlights. Presets in [`style::elevation`](../../src/ui/style.rs) (`LOW/MEDIUM/HIGH`, `INNER_*`, `BORDER_HIGHLIGHT*`, `SURFACE_HIGHLIGHT*`). |

---

## Hero reference surface

The **canonical composition** for margins and stroke tiers is **header (tab bar + instrument rule) → main chat viewport → composer deck**. New screens should match this rhythm before inventing new gutters.

| Element | Token / constant | Notes |
|---------|------------------|--------|
| Main viewport + composer horizontal gutter | `style::hero::MAIN_VIEWPORT_GUTTER` | Poster gutter between window edge and content/composer; same value for vertical top inset above message list. |
| Header instrument rule | `stroke::INSTRUMENT_RULE_PX` | Thick cream rule; tab bar sits above it. |
| Tab bar | `hero::TAB_BAR_WIDTH`, `hero::TAB_BAR_HEIGHT` | Centered folder in the header band. |
| Tab chrome strokes | `stroke::TAB_BAR_RING_PX`, nested insets | Nested shell → well → track. |
| Composer chassis | `chrome::COMPOSER_BACKPLATE`, `stroke::COMPOSER_RULE_PX` | Deck read vs instrument rule. |

---

## Motion — tape-deck / mechanical feel

The UI uses spring integration in [`src/utils/animation.rs`](../../src/utils/animation.rs). **Preferred language** for chrome and navigation:

- **Preset:** `AnimationPreset::Mechanical` — higher damping relative to stiffness, little or no overshoot; reads as mass and friction, not a playful bounce.
- **Tab slider:** leading and trailing edge animations use **Mechanical** (not Bouncy / TightBounce) so the active tab pill settles like a latch, not a rubber band.
- **When to use springs with overshoot:** Cursor or micro-interactions that benefit from a tiny bounded overshoot may use `TightBounce` with `max_bounce`; avoid for panels, tabs, and mode switches.
- **Duration:** There is no fixed global duration—springs run until `is_at_target`. Tune stiffness/damping so typical UI motion **settles within roughly 200–400 ms** at 60 Hz for Mechanical presets on positional UI.

For liquid or trailing-edge stretch (e.g. a deliberate viscous trailing edge), `Viscous` remains available but should be the exception, not the default for mission-control chrome.

---

## Tactile — depth and feedback (practice)

- **Shadows / elevation:** A single off-screen **key light at `-45°`** (upper-left) drives the whole scene. Every lighting primitive — outer drop shadow, inner drop shadow, specular border rim, specular surface sheen — shares that direction so the whole UI reads as one consistent room, not per-component glows.
- **Primitive set:**
  - [`ShadowSpec`](../../src/ui/shadow.rs) + [`Renderer::queue_shadow`](../../src/gfx/renderer.rs) — outer drop shadow at `+135°`, feathered gradient. Presets: `elevation::{LOW, MEDIUM, HIGH}`. Called *before* the component fill.
  - [`InnerShadowSpec`](../../src/ui/shadow.rs) + `Renderer::queue_inner_shadow` — shadow on the top-left interior (light-facing inner wall). Presets: `elevation::{INNER_LOW, INNER_MEDIUM, INNER_HIGH}`. Called *after* the fill; shader clips to the rounded shape.
  - [`BorderHighlightSpec`](../../src/ui/shadow.rs) + `Renderer::queue_border_highlight` — bright rim on the top-left inner edge, modulated by an SDF-gradient Lambert term. Presets: `elevation::{BORDER_HIGHLIGHT, BORDER_HIGHLIGHT_STRONG}`. Called *after* the fill.
  - [`SurfaceHighlightSpec`](../../src/ui/shadow.rs) + `Renderer::queue_surface_highlight` — diagonal sheen across the interior, brightest at top-left. Presets: `elevation::{SURFACE_HIGHLIGHT, SURFACE_HIGHLIGHT_STRONG}`. Called *after* the fill.
- **Implementation map:** [`style::elevation::SHADOW_SIZE_PX`](../../src/ui/style.rs) sets the shared offset magnitude; the internal `directional(sigma, alpha)` helper builds a `ShadowSpec` with offset `(SHADOW_SIZE_PX / √2, SHADOW_SIZE_PX / √2)`. Tiers pick (sigma, alpha) pairs. Highlight colors are warm-white, pulled slightly toward `text::PRIMARY` (never pure white). Each primitive emits **one quad** routed through the UI pipeline. Fragment shader branches live in [`ui_shader.wgsl`](../../src/gfx/shaders/ui_shader.wgsl):
  - `bubble == 3.0` — `0.5 - 0.5 · erf(sdf / (σ · √2))` of the rounded-box SDF (Abramowitz & Stegun `erf7`).
  - `bubble == 4.0` — same `erf` formulation but on the *offset-inward* SDF, clipped to the real shape.
  - `bubble == 5.0` — reconstructs surface normal via central-difference of the rounded-box SDF, then multiplies Lambert (`max(0, dot(-L, n))`) by a gaussian rim pulse peaking at the inside edge.
  - `bubble == 6.0` — diagonal falloff `pow(1 - (u+v)/2, curve)` across the interior, clipped to the shape.
  - All branches are O(1) per pixel — no extra render pass, no extra texture.
- **Currently lit:** header, tab shank+cap, sidebar, composer chassis, shard backplates (all use outer drop shadow). Inner shadow / border highlight / surface highlight primitives are available for opt-in per component.
- **Stateful chrome:** Button `Pressed` / `Hover` fills, tab slider position, sidebar selection rings, and graph node/shard state should **persist** on screen so the workspace feels inhabited.
- **DnD and transforms:** Any repositioning interaction should expose **intermediate visuals** (semi-transparent follower, snap guides, invalid-drop tint) until we commit layout.
- **Alignment with cassette futurism:** Tangibility reads as **lab hardware**—detents, bezels, labeled modules—not glossy app-store cards.

---

## Readout typography and markdown

| Role | Token |
|------|--------|
| Inline code / fenced code foreground (on dark panels) | `style::markdown::CODE_FOREGROUND` |
| Same-word highlight (stylus / editor) | `style::editor::MATCH_HIGHLIGHT_TEXT` |
| Text selection band under caret | `style::editor::SELECTION_BAND` |
| Pinned graph icon tint | `style::graph::PIN_ICON_TINT` |

Body and titles elsewhere stay on `style::font_size::*` and `style::text::*` as already used in windows and modals.

---

## Atmosphere (glow, grain, depth)

- **Single layer:** Keep phosphor glow / parallax / vignette in the background pass (`backdrop::*`, glow component). Do not stack multiple full-screen “CRT” effects.
- **Low amplitude:** Haze and bands stay subtle so `text::PRIMARY` on `bg::*` stays the dominant readout.
- **No cool gray code:** Code and monospace readouts use warm `markdown::CODE_FOREGROUND`, not neutral LCD gray.

---

## Empty states

When a viewport has no primary content, use a **quiet field** plus short copy:

- Horizontal inset: `style::empty_state::SIDE_INSET` (same family as `hero::MAIN_VIEWPORT_GUTTER`).
- Title: `empty_state::TITLE_FONT`, subtitle: `empty_state::SUBTITLE_FONT`, gap: `empty_state::VERTICAL_GAP`.
- Center the block vertically in the remaining rect or align top with **double** the side inset from the first baseline.

Optional geometric marks stay small and flat (see Part 2 — simple geometric marks).

---

## Transient UI (toasts, modals)

| Piece | Token / behavior |
|-------|------------------|
| Toast card | `style::toast::*` (width, padding, stack gap, margins); rim `border::INSTRUMENT_RULE`; panel `bg::PANEL_POPUP`; stripe by type: Error `accent::WARNING`, Success `accent::PHOSPHOR`, Info `accent::POP_COOL`. |
| Modal backdrop | Tinted bulkhead: `bg::PRIMARY` RGB × caller opacity (not pure black). |
| Modal panel shell | Outer rim quad `border::INSTRUMENT_RULE` + `stroke::COMPOSER_RULE_PX` inset; inner fill `bg::PRIMARY`, radius `corner_radius::LARGE`. |

---

## Sidebar and formatting toolbar

| Piece | Token |
|-------|--------|
| Sidebar section title height, row height, section gap, title button padding | `style::sidebar_layout::*` |
| Notepad toolbar bar height, button size, spacing | `style::toolbar_chrome::*` |

Interaction rule for sidebar/library list cards: when a row uses expandable management actions, action controls must use the shared icon vocabulary (`DOTS_6_VERTICAL` handle, `PENCIL` rename/edit, `TRASH` delete) rather than letter labels or ad-hoc glyphs. This preserves cross-pane affordance consistency and keeps management semantics instantly recognizable.
Library main-pane management toolbar actions follow the same icon-first rule (`TRASH` delete, `PLUS` add to collection, `CLOSE` remove from collection) rather than text labels.

---

## Theme presets (Settings → Theme)

Full palettes live in [`src/ui/theme.rs`](../../src/ui/theme.rs) (`ThemePalette`, `THEME_CHOICES`). **Moodboard experiments** (iterate in code there):

| Id | Intent |
|----|--------|
| `standard` | Default cassette (blue-violet deck, cream readout). |
| `cassette-sage` | Charcoal + sage wells, **burnt orange** user / impulse, **teal** cool accent, taupe secondaries, warm/teal parallax haze. |
| `cassette-sunburst` | Umber black, **mustard** phosphor, orange–teal stripe haze, warm composer paper. |
| `cassette-vector` | Near-black HUD, **green phosphor** text family, **amber** pop, **cyan** data accent. |
| `cassette-field` | Exact swatches: bg `#242623`, header deck `#525B60`, text `#CAC977`, user bubble `#DC4C0D`, trim/accent `#71BDBD`. |

Structural stroke widths (`style::stroke::*` — instrument rule, composer rule, tab folder ring, bubble rims, graph edges) are **~6–8px** for thick CRT-style bezels app-wide.

Switch in-app under **Settings**, or set `"theme"` in `data/settings.json` to one of the ids above.
