# Host logodetect on aws ec2

1. start ubuntu 18.04 deep learning ami on a t2.medium instance and attach a 90GB harddrive.
2. configure your `~/.ssh/config` to have an `aws_logodetect` host pointing to your instance.
3. run `sh ./aws_scp.sh` to copy all files to the ec2 instance (make sure to have models and data in the root of this repo first)
4. connect to the machine with `ssh aws_logodetect`.
5. run `sh ./aws_install.sh` on the machine to install the library, dependencies, and configure apache to run the app.