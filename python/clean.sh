rm -rf data/reinforce
mkdir data/reinforce
mkdir model/reinforce/alphanet-black
mkdir model/reinforce/alphanet-white
cp model/reinforce/alphanet-5-black/alphanet-5-black-0.data-00000-of-00001 model/reinforce/alphanet-black/
cp model/reinforce/alphanet-5-black/alphanet-5-black-0.index model/reinforce/alphanet-black/
cp model/reinforce/alphanet-5-black/alphanet-5-black.meta model/reinforce/alphanet-black/
cp model/reinforce/alphanet-5-white/alphanet-5-white-0.data-00000-of-00001 model/reinforce/alphanet-white/
cp model/reinforce/alphanet-5-white/alphanet-5-white-0.index model/reinforce/alphanet-white/
cp model/reinforce/alphanet-5-white/alphanet-5-white.meta model/reinforce/alphanet-white/
echo -e "model_checkpoint_path: \"alphanet-5-black-0\"\nall_model_checkpoint_paths: \"alphanet-5-black-0\"" > model/reinforce/alphanet-black/checkpoint
echo -e "model_checkpoint_path: \"alphanet-5-white-0\"\nall_model_checkpoint_paths: \"alphanet-5-white-0\"" > model/reinforce/alphanet-white/checkpoint
rm -rf model/reinforce/alphanet-5-black
rm -rf model/reinforce/alphanet-5-white
mv model/reinforce/alphanet-black model/reinforce/alphanet-5-black
mv model/reinforce/alphanet-white model/reinforce/alphanet-5-white
