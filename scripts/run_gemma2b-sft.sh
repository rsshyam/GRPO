python -u train.py model=gemma-2b datasets=[goqa_0,goqa_5] train_frac=0.8 loss=sft gradient_accumulation_steps=2 batch_size=16 eval_batch_size=8 sample_during_eval=False trainer=GroupTrainer lr=1e-4 eval_every=192 eval_train_every=192