def log_evaluation(logpath, text):
    with open(logpath, 'a+') as fout:
        fout.write(text)
