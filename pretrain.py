import random

import torch

from utils import pprint, set_gpu, get_command_line_parser, postprocess_args
from model.trainer.pretrainer import PreTrainer
import random

if __name__ == "__main__":
    torch.manual_seed(int(random.random()*100000))
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    pprint(vars(args))
    set_gpu(args.gpu)
    trainer = PreTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)
