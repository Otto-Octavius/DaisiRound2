# Attention Heatmaps by Self-Supervised Transformer


## Requirements

Both the Daisi App and the API  the API calls from notebooks/Py-scripts doesn't require any libraries.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PIL

```bash
pip install pydaisi
```

## Calling
Get a string containing a vague address consisting of the building, location or the city which it's present in.

```python
import pydaisi as pyd
i = input("Paste the absolute/env address of the video sample: ")
Daisi = pyd.Daisi("sam-dj/Attention Heatmaps by Self-Supervised Transformer")
```

## Passing and Rendering
We simply pass the string to the Daisi and finally save the result

```python
Daisi.generate(i)
```

## Running the Daisi App

As mentioned earlier, this can be automated just by [Running the Daisi App](https://app.daisi.io/daisies/sam-dj/Attention%20Heatmaps%20by%20Self-Supervised%20Transformer/info)

## References
My notebook which attempts to implement these ideas [Sam DJ's Notebook](https://colab.research.google.com/drive/1ULHrSrhBz5kQbZ6hRacCy2_j_on5CSKn?usp=sharing)

The official article of [DINO and PAWS](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/)

The official paper of [DINO](https://arxiv.org/pdf/2104.14294.pdf)
