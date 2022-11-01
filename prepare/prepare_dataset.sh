file_path=$(pwd)
data_path=$1
echo "file_path: $file_path"
echo "data_path: $data_path"

mkdir -pv $data_path/ImageNet-2012
cd $data_path/ImageNet-2012

path=$(pwd)
echo "Now I am at: $path"
# download dataset
wget -cnv https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget -cnv https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget -cnv https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar --no-check-certificate
wget -cnv https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz --no-check-certificate

touch ILSVRC2012_img_train.tar
touch ILSVRC2012_img_val.tar
touch ILSVRC2012_img_test_v10102019.tar
touch ILSVRC2012_devkit_t3.tar.gz
mv ILSVRC2012_devkit_t3.tar.gz devkit.tar.gz

# make train directory and unzip files
mkdir -pv train
tar --touch -xvf ILSVRC2012_img_train.tar -C train
# prepare train data
cd train
for file in *.tar;do
   filename=$(basename $file .tar)
   if [ ! -d $filename ];then
       mkdir -pv $filename
   else
       rm -rf $filename
   fi
   tar --touch -xvf $file -C $filename
   rm $file
done

# Maek val directory and unzip files
cd $data_path/ImageNet-2012
mkdir val
tar --touch -xvf ILSVRC2012_img_val.tar -C val/
# prepare val data
cd $file_path/prepare
bash val_prepare.sh $data_path/ImageNet-2012/val

# resize dataset
python $file_path/prepare/resize.py --data_path $data_path