# agent streaming examples


# Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

### TEST AGENT GRAPH ###

# # WORKING STREAM - test 2 - properly configured for AIMessageChunk streaming
# for chunk in react_graph.stream(
#         {"messages": [HumanMessage(content="what is day trading?")]},
#         stream_mode="messages"):
#     # chunk is a tuple: (node_name, message_data)
#     message, metadata = chunk
#     print(f"Node: {metadata["langgraph_node"]}")
#     print(f"Message content: {message.content}")
#     print(f"Message type: {type(message)}")
#     print("---")

# test 1
# messages = react_graph.invoke(
#     {"messages": [HumanMessage(content="Should I invest in Tesla stocks?")]})
# for message in messages['messages']:
#     message.pretty_print()

# test 2 - invoke - WORKS
# messages = react_graph.invoke(
#     {"messages": [HumanMessage(content="what is day trading?")]})
# for message in messages['messages']:
#     message.pretty_print()

# For clean content-only streaming:
# print("\n=== CLEAN CONTENT STREAMING ===")
# for chunk in react_graph.stream(
#         {"messages": [HumanMessage(content="what is day trading?")]},
#         stream_mode="messages"):
#     node_name, message_data = chunk
#     if hasattr(message_data, 'content') and message_data.content:
#         print(f"[{node_name}]: {message_data.content}")

# For real-time token streaming (if the LLM supports it):
# print("\n=== REAL-TIME TOKEN STREAMING ===")
# for chunk in react_graph.stream(
#         {"messages": [HumanMessage(content="what is day trading?")]},
#         stream_mode="messages"):
#     node_name, message_data = chunk
#     if hasattr(message_data, 'content') and message_data.content:
#         print(message_data.content, end="", flush=True)
# print()  # Final newline
