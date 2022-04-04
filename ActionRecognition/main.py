import wget

url1 = 'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip'

url2 = 'http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip'
filename1 = wget.download(url1)
filename2 = wget.download(url2)

filename1
filename2


print('Hello')