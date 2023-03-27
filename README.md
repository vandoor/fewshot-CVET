如果要运行pretrain:

```shell
python pretrain.py --max_epoch 200 --episodes_per_epoch 1000 --model_class PreMod --use_euclidean --backbone_class Res18 --D 512  --dataset cifar100 --num_classes 100 --way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --init_lr 0.00001 --lr_scheduler step --init_weights xxx --eval_interval 1 --step_size 20 --gamma 0.5 --beta 0.01 --batch_size 10 --save_dir yyy
```
其中--init_weight代表加载预训练模型文件的位置，--save_dir表示存放结果的根目录