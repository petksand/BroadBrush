filepaths = []
path = "artist_dataset/"
eras = os.listdir(path)
if ".DS_Store" in eras:
    eras.remove(".DS_Store")
for era in eras:
    artists = os.listdir("{}{}/".format(path,era))
    if ".DS_Store" in artists:
        artists.remove(".DS_Store")
    for artist in artists:
        pics = os.listdir("{}{}/{}/".format(path,era,artist))
        if ".DS_Store" in pics:
            pics.remove(".DS_Store")
        for i, pic in enumerate(pics):
            pic_path = "{}{}/{}/{}".format(path, era, artist,pic)
            filepaths.append(pic_path)
print(filepaths)
filepaths = np.asarray(filepaths)
np.savetxt("filepaths.csv", filepaths, delimiter=",", fmt="%s")