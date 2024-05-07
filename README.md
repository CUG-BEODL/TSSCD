<p align="center">

<img src="img/TSSCD.png" width="200">

</p>
<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Time-series land cover change detection using deep learning-based temporal semantic segmentation </b></h2>
</div>

<div align="center">

**[[Paper Page]](https://doi.org/10.1016/j.rse.2024.114101)**



</div>



### What is TSSCD?

1. We were inspired by the semantic segmentation task of remote sensing images and defined land cover types as semantics. However, we implemented temporal semantic segmentation on a one-dimensional time series. Land cover changes in the representation of remote sensing time series were considered semantic boundaries, and we used existing mature semantic segmentation models to segment different time intervals.
2. To meet the demand for more complex remote sensing time series interpretation and to bridge the **"semantic gap"** between low-level **temporal changes** and high-level **semantic changes**, we build an end-to-end change detection model based on temporal semantic segmentation in the field of computer vision as a prototype design.


<p align="center">

<img src="img/flow_new.png">

</p>

### Quick view

You can run demo.py to visualize the output of the TSSCD model. The trained model is saved under the **model_data** folder. Please note that this model parameters is only valid in the research area specified in the paper, and its generalization performance has not been evaluated.

```
python demo.py
```


The input data are all 10m and 20m bands (10 in total) of Sentinel-2, and the time resolution is one month.
![img.png](img/img.png)

### Retrain TSSCD

#### Note:
1. Prepare your training samples and time-phase land cover labels, pay attention to modify the **data_loader.py**. For detailed data format, please refer to the dataset folder.
2. Samples can be obtained by accessing the link：[https://figshare.com/articles/dataset/Urban_monthly_land_dynamics_Sentinel-2_benchmark_dataset/25764108](https://figshare.com/articles/dataset/Urban_monthly_land_dynamics_Sentinel-2_benchmark_dataset/25764108)
3. Retrain TSSCD with the following command, and the model with the highest change detection and classification accuracy will be automatically saved.
4. We strongly recommend using the data set construction method mentioned in Section 5.2 of the paper. If you have a better annotation method or annotation tool, please feel free to contact us.

```
python run.py
```

### Contact

> 🙋 If you have any question or want to use the code, please contact yanjn@cug.edu.cn.

> 🌟 If you find this resource helpful, please consider to star this repository and cite our research.

> 🌍 If you want to view the study area mapping online, you can visit ([Google Earth Engine](https://ee-hhx.projects.earthengine.app/view/tsscd))

```
@article{He2023Timeseries,
title = {Time-series land cover change detection using deep learning-based temporal semantic segmentation},
journal = {Remote Sensing of Environment},
volume = {305},
pages = {114101},
year = {2024},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2024.114101}
}
```

### Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

[https://github.com/repository-preservation/lcmap-pyccd](https://github.com/repository-preservation/lcmap-pyccd)

[https://github.com/mortvest/bfast-py/tree/master](https://github.com/mortvest/bfast-py/tree/master)


