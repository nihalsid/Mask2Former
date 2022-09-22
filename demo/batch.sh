scenes=("room_2" "office_0" "office_2" "office_3" "office_4")

for scene in ${scenes[@]}; do
  python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --n $1 --p $2 --input data/replica/${scene}/color --output data/replica/${scene}/panoptic --predictions data/replica/${scene}/panoptic --opts MODEL.WEIGHTS ../checkpoints/model_final_f07440.pkl
done