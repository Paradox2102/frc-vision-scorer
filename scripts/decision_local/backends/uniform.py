"""Plumbing check backend: uniform probabilities, no model."""


class Backend:
    name = "uniform-test"

    def __init__(self, args):
        pass

    def answer(self, state, images, questions):
        assert images, "no image decoded"
        return {"answers": {qid: {"probabilities": {o: 1.0 for o in q["criteria"]}}
                            for qid, q in questions.items()},
                "input_tokens": None}
