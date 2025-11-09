from sel import detect_user_intent


def test_intent_greeting():
    assert detect_user_intent("Hey Sel, you around?") == "greeting"


def test_intent_support():
    assert detect_user_intent("I'm feeling really anxious tonight.") == "support"


def test_intent_technical():
    assert detect_user_intent("Can you help debug this python traceback?") == "technical"


def test_intent_identity():
    assert detect_user_intent("Who are you really?") == "identity"


def test_intent_question_fallback():
    assert detect_user_intent("What do you think about art?") == "question"
