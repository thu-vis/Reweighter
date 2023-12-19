# Reweighter: Interactive Reweighting for Mitigating Label Quality Issues

## Install (Docker)

To run the container:
```sh
sudo docker run -p 49623:49623 -t vicayang/reweighter:latest /bin/bash /autorun.sh
```
Then open `http://localhost:49623/` in your browser.

This should reproduce the Fig. 9 in our paper (Fig. 9 is slightly different from the screenshot because we adjust the aspect ratio and add some notations to make the figure more readable in the paper).

## Install

1. This project uses [python 3.7](https://www.python.org/). Go check it out if you don't have it installed.

2. Install python package.
```sh
pip install -r requirements.txt
```

3. Install nodejs package: check `README.md` under `vis` for more details.

4. Download the data ([link](https://drive.google.com/file/d/12db1lcp1GjG0ujCNE-nt-B5racen-pJ_/view?usp=sharing)), unzip it, and put it in `flask/application/data`.

## Run

1. run backend. 
```sh
cd flask; python app.py
```

2. run frontend: check `README.md` under `vis` for more details.


If you encounter any issues when using this library, please feel free to contact me vicayang496@gmail.com.