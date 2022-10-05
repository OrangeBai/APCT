file_path=$(pwd)
data_path=$1
echo "file_path: $file_path"
echo "data_path: $data_path"

mkdir -pv $data_path/ImageNet-2012
cd $data_path/ImageNet-2012


path=$(pwd)
echo "Now I am at: $path"

wget -c image-net-train-url --no-check-certificate
wget -c image-net-val-url   --no-check-certificate
wget -c image-net-test-url  --no-check-certificate
wget -c image-net-kit-rul   --no-check-certificate

touch ILSVRC2012_img_train.tar
touch ILSVRC2012_img_val.tar
touch ILSVRC2012_img_test_v10102019.tar
touch ILSVRC2012_devkit_t3.tar.gz
mkdir val
mv ILSVRC2011_devkit-2.0 devkit


mkdir train
tar --touch -xvf ILSVRC2012_img_train.tar -C train

cd train
for file in *.tar;do
   filename=$(basename $file .tar)
   if [ ! -d $filename ];then
       mkdir $filename
   else
       rm -rf $filename
   fi
   tar --touch -xvf $file -C $filename
   rm $file
done

cd $data_path/ImageNet-2012

tar --touch -xvf ILSVRC2011_devkit-2.0.tar.gz -C val/

cd $file_path/prepare
bash val_prepare.sh $data_path/ImageNet-2012/val
python $file_path/prepare/resize.py --data_path $data_path