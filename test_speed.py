from nanochat.tokenizer import RustBPETokenizer


def legacy_encode(tok, text, prepend=None, append=None, num_threads=8):
    # text can be either a string or a list of strings

    if prepend is not None:
        prepend_id = prepend if isinstance(prepend, int) else tok.encode_special(prepend)
    if append is not None:
        append_id = append if isinstance(append, int) else tok.encode_special(append)

    if isinstance(text, str):
        ids = tok.enc.encode_ordinary(text)
        if prepend is not None:
            ids = [prepend_id, *ids]
        if append is not None:
            ids.append(append_id)
        ids = list(ids)
    elif isinstance(text, list):
        ids = tok.enc.encode_ordinary_batch(text, num_threads=num_threads)
        if prepend is not None:
            for ids_row in ids:
                ids_row = [prepend_id, *ids_row]
        if append is not None:
            for ids_row in ids:
                ids_row.append(append_id)
        ids = [list(row) for row in ids]
    else:
        raise ValueError(f"Invalid input type: {type(text)}")

    return ids


tokenizer = RustBPETokenizer.from_pretrained("gpt2")


# Test speed of new vs legacy tokenizer
def test_speed():
    import time

    texts = ["Hello, how are you doing today? I hope you're having a great day!" for _ in range(1000)]

    # Warm up
    for _ in range(10):
        tokenizer.encode(texts, prepend="<|endoftext|>", num_threads=4)
        legacy_encode(tokenizer, texts, prepend="<|endoftext|>", num_threads=4)

    # Test new tokenizer
    start_time = time.time()
    for _ in range(100):
        tokenizer.encode(texts, prepend="<|endoftext|>", num_threads=4)
    new_duration = time.time() - start_time

    # Test legacy tokenizer
    start_time = time.time()
    for _ in range(100):
        legacy_encode(tokenizer, texts, prepend="<|endoftext|>", num_threads=4)
    legacy_duration = time.time() - start_time

    # print(f"New tokenizer duration: {new_duration:.4f} seconds")
    # print(f"Legacy tokenizer duration: {legacy_duration:.4f} seconds")

    return new_duration, legacy_duration

if __name__ == "__main__":
    trial_num = 10
    new_duration_avg = 0.0
    legacy_duration_avg = 0.0
    for _ in range(trial_num):
        print("Running trial", _+1)
        new_duration, legacy_duration = test_speed()
        new_duration_avg += new_duration / trial_num
        legacy_duration_avg += legacy_duration / trial_num
    print(f"Average new tokenizer duration over {trial_num} trials: {new_duration_avg:.4f} seconds")
    print(f"Average legacy tokenizer duration over {trial_num} trials: {legacy_duration_avg:.4f} seconds")