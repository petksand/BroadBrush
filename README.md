# BroadBrush
## An Art Classifier

### What is it?
BroadBrush is an art classifier. Given a piece of art, BroadBrush will determine the artist and era from which it came.

BroadBrush is trained on 6 eras and 24 artists, summarized in the table below.

| Era | Artists|
|-----|------|
| Baroque | Annibale Carracci, Caravaggio, Diego Velasquez, Rembrandt |
| Cubism | Fernand Leger, Juan Gris, Louis Marcoussis, Marc Chagall|
| Impressionism | Mary Cassat, Joaquin Sorolla, Edgar Degas, Edouard Manet|
| Pop Art | Andy Warhol, Hiro Yamagata, Patrick Caulfield, Roy Lichtenstein|
| Realism | Anders Zorn, Boris Kustodiev, Henri Fantin, Camille Carot|
| Renaissance | Albrecht Altdorfer, Hans Baldung, Lorenzo Lotto, Michelangelo|

### How to Use It
To see how well the model performs on your artwork of choice, follow these steps:
1. Select any image from an artist in the table above.
2. Crop the image to 300x300 pixels.
3. Place the image in a folder with the artist's name. The artists's name should appear as it does in the table above, all lowercase, and with underscores in place of spaces: Louis Marcoussis -> louis_marcoussis. (The model will know which era it belongs to by the artist alone, so you won't need to specify it).
4.  Place that folder into a folder titled "artist_dataset_train"
5.  Run the code with the following terminal command: `python artist_bot.py`. The code will print the accuracy as it runs, then print the final accuracies once it terminates.
6.  To see the confusion matrices it produced, simply run `python conf.py` and the matrices will pop up.

You may repeat steps 1-4 to build your own test data set and run it on our model.
