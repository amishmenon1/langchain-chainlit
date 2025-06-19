import chainlit as cl


async def update_message(msg: cl.Message, content: str):
    msg.content = content
    await msg.update()
    return msg


async def new_message(content: str):
    new_msg = await cl.Message(content=content).send()
    await new_msg.update()
    return new_msg
