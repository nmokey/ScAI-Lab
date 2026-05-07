import json


def collect_mc_answers_from_llm_output(answer_file, save_file, pred_key="model_answer"):
    """No-op for open-ended generation — retained for interface compatibility."""
    with open(answer_file, "r") as f:
        questions = json.load(f)
    with open(save_file, "w") as f:
        json.dump(questions, f, indent=4)
