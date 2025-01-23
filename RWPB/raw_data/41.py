def merge_short_sentences_en(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    # ----

    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

# unit test cases
print(merge_short_sentences_en(["I am.", "Here now.", "John is going to the store.", "Oh.", "Really?", "That's interesting."]))
print(merge_short_sentences_en(["This is a proper sentence.", "Here is another example.", "Each of these sentences is long enough."]))
print(merge_short_sentences_en(["Weather is bad.", "Indeed.", "Very cold."]))