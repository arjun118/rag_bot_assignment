from aiogram.fsm.state import State, StatesGroup


class ChatState(StatesGroup):
    waiting_for_image = State()
    waiting_for_rag_query = State()
