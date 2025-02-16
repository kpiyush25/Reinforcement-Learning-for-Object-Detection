# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export HOME="/storage/home/sidnayak"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py35
# Change to the directory in which your code is present
cd /storage/home/sidnayak/Reinforcement-Learning-for-Object-Detection/src_phase2/
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
# python -u Agent_detect.py --load_model=1 --epoch=10 &> out
python -u ssd_random.py --epoch=20 --low=0.1 --high=1.9 &> outputs/out_random_newReward_1
python -u ssd_random.py --epoch=20 --low=0.4 --high=1.6 &> outputs/out_random_newReward_2
python -u ssd_random.py --epoch=20 --low=0.5 --high=1.5 &> outputs/out_random_newReward_3

