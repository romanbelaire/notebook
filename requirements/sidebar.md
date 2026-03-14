user story: new chat
- I click the new chat button, or press ctrl+n, and a new chat is opened and the chat bar is autofocused.
- I click on other chats in the sidebar. They load when I click on them. the information modal does not open.
- there is a handle on the right side of the chat button, for every row. when clicked, it expands into two buttons, 'delete' and 'info'. moving  the mouse off of the sidebar collapses this back into the handle
- the handle will be draggable in the future.
the new chat button is right-aligned at the horizontal level of the section header.
these behaviors will be the same for all items in the sidebar.

story: send message
- I type and send a message. I recieve a response from the llm if successful, otherwise I get a toast with the error
note: at current time, the API console is not showing any endpoint hit when I send a message.
- The the messages are contained in their own chat bubbles, with the user's on the right hand side of the screen and the LLM's on the left.
- At the bottom of the bubble is a handle that expands into edit, hide, delete, and pin buttons. the behavior is detailed in the original react code.
- The user message and agent reply are two parts of a shard.

story: window controls
- I should be able to drag, resize, move, close, minimize, maximize the window.

concept: shards
a shard is a blob of context, essentially, and is the core first-class citizen in this data ecosystem. 
it is initially created from a message. By _message_, we include both the user message and the LLM response.
a shard is a bit of context that gets sent to the LLM, so it obviously has text. but, it also has:
- vector embedding
- list of sources
    - PDFs, links, friend shards
- a parent (can be none)
    - the message that preceeded this one
- a list of children
    - any number of messages spawned directly from this one
- a list of friends
    - other shards that were linked in by the user as additional context, but were not the parent shard.

for now, the chat window will just have one linear thread of shards as a chat object, but we will later expand this to _constellations_, or essentially knowledge graphs between shards.
