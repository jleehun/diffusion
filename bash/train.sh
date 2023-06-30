
python /root/diffusion/simple-diffusion/train.py   
    # --dataset-name=cifar10 \
    # --resolution=32 \
    # --output_dir=trained_models/ddpm_cifar.pth \
    # --train_batch_size=4 \
    # --num_epochs=121 \
    # --gradient_accumulation_steps=1 \    
    # --learning_rate=1e-4 \
    # --lr_warmup_steps=300

# python ../simple-diffusion.train.py \
# 	--dataset-name ImageNET \
#     --train-data-path \
#     --output-dir \
#     -- \
#     -- samples-dir\
#     -- \
#     -- \
#     -- \
#     -- \
#     -- \
#     -- \
#     -- \