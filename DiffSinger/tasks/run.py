import importlib
from utils.hparams import set_hparams, hparams
import torch
torch.cuda.empty_cache()

def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    # print('cls_name', cls_name) # FastSpeech2Task | DiffSingerMIDITask
    task_cls = getattr(importlib.import_module(pkg), cls_name) 
    # tasks.tts.fs2.FastSpeech2Task | usr.diffsinger_task.DiffSingerMIDITask
    # print("| Task: ", task_cls) 
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
