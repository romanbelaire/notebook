- window resizing

- blinking cursor

- smooth interpolation for cursor movement

- parallax background

- more polished elements, i.e. rounded corners, standardized shapes and visual lanugage
    - should we achieve this with some kind of design config file?

- text rendering bugs:
    - chat: on editing message, the text is not constrained to the message box. should constrain the width of the text, and expand/contract the box vertically to accomodate as the user writes.
    - notepad: cursor does not follow text properly. it is still using the monospace guessing instead of glyph positions. 
    - notepad (and probably elsewhere): text formatting like bold, italic etc doesnt work
    - make sure the chat bubbles support markdown rendering
    - implement the highlighting function: whenever a word is highlighted, other instances of this word also get highlighted. we have this behavior in the original react version, but slightly different. I think we colored the text when bolded or something.

- closing sidebar should keep the chat the same width, just moved over to the center.