# Fair-and-Interpretable-Plant-Disease-Detection
Creating a Fair and Interprestable Plant Disease Detection Model which takes in RGB image as an input and maps this into a Hyperspectral Image (HSI) and then use this HSI to Classify the disease and as well as provide a description (Reseaoning for the label classified). 
## Work Package 1 (Data Collection):
### Plant Disease Datasets: 
| Dataset | In the wild? | #Plants | #Classes | #Images | Labels? | Description? | Segmentation Mask? |
|---|---|---|---|---|---|---|---|
| Plant Village[1] | No | 14 | 38 | 87K | Yes | No | No |
| Plant Seg[2] | Yes | 34 | 115 | 11k | Yes | No | Yes |
| Plant Doc[3] | Yes | 13 | 27 | 2.5K | Yes | No | No |
| Plant Wild[4] | Yes | 33 | 89 | 18.5k | Yes | Yes | No | 

Note: Use these datasets as benchmarks for comparison.

### Custom Dataset:
Create an Agriculture Expert Annotated dataset which contains:
1. RGB Images (with Segmentation mask + Labels + Description)
2. HyperSpectral Images (with Segmentation mask + Labels + Description

### References:
1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016, September). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419. [[Github]](https://github.com/spMohanty/PlantVillage-Dataset).
2.  Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P., & Huang, Z. (2024). PlantSeg: A large-scale in-the-wild dataset for plant disease segmentation. arXiv preprint arXiv:2409.04038. [[Github]](https://github.com/tqwei05/PlantSeg) [[Paper]](https://arxiv.org/abs/2409.04038).
3.  Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. In Proceedings of the 7th ACM IKDD CoDS and 25th COMAD (pp. 249–253). Association for Computing Machinery. [[Github]](https://github.com/pratikkayal/PlantDoc-Dataset?tab=readme-ov-file) [[Paper]](https://arxiv.org/abs/1911.10317).
4.  Wei, T., Chen, Z., Huang, Z., & Yu, X. (2024). Benchmarking in-the-wild multimodal plant disease recognition and a versatile baseline. In Proceedings of the ACM International Conference on Multimedia. [[Github]](https://github.com/tqwei05/MVPDR) [[HuggingFace]](https://huggingface.co/datasets/uqtwei2/PlantWild) [[Website]](https://tqwei05.github.io/PlantWild/) [[Paper]](https://arxiv.org/abs/2408.03120).

## Work Package 2 (RGB based Plant Classification Model):
### Recent Work done:
1. Ghosh, S., Singh, A., Kavita, Jhanjhi, N. Z., Masud, M., & Aljahdali, S. (2022). SVM and KNN based CNN architectures for plant classification. Computers, Materials & Continua, 71(3), Article 46503. [[Paper]](https://www.techscience.com/cmc/v71n3/46503/html).

## Work Package 3 (RGB -> HSI Mapping):
### Datasets:
RGB-HSI Pair available datasets
#### Open-Source:
| Dataset    | Image Count | Resolution        | Spectral Bands | Step      | Wavelength Covered   | Scenes Covered                  | Remarks                               |
|------------|-------------|-------------------|----------------|-----------|----------------------|---------------------------------|---------------------------------------|
| CAVE[1]       | -           | 512 × 512         | 31             | 10 nm     | 400 nm–700 nm        | 32 scenes, various controlled indoor objects | PNG image type |
| ICVL[2]       | 200         | 1392 × 1300       | 519            | 1.25 nm   | 400 nm–1000 nm       | Everyday indoor & outdoor scenes | Downsampled 31 bands dataset of the same is also given |
| KAUST‑HS[3]   | 409         | 512 × 512         | 34             | -         | 400 nm–730 nm         | Indoor and outdoor scenes      | Only contains HSI images, have to extract RGB from them first |


#### NTIRE Datasets (Availability Status):

### Dataset References:
1. Cave [[Dataset]](https://cave.cs.columbia.edu/repository/Multispectral)
2. ICVL [[Dataset]](https://huggingface.co/datasets/danaroth/icvl)
3. KAUST-HS [[Dataset]](https://repository.kaust.edu.sa/items/891485b4-11d2-4dfc-a4a6-69a4912c05f1)

### Recent Work (Algorithms and mappings):
1. Yuanhao Cai, Jing Lin, Zudi Lin, Haoqian Wang, Yulun Zhang, Hanspeter Pfister, Radu Timofte, and Luc Van Gool. **Mst++**: Multi-stage spectral-wise transformer for efficient spectral reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [[Paper]](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2204.07908) [[Code]](https://github.com/caiyuanhao1998/MST-plus-plus)
2. Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. **Restormer**: Efficient transformer for high-resolution image restoration. In CVPR, 2022. [[Paper]](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf) [Code](https://github.com/swz30/Restormer).
3. Ping Wang and Xin Yuan. Saunet: Spatial-attention unfolding network for image compressive sensing. In Proceedings of the 31st ACM International Conference on Multimedia. [[Paper]](https://dl.acm.org/doi/10.1145/3581783.3612242) [[Code]](https://github.com/pwangcs/SAUNet).
4. Zhiyang Yao, Shuyang Liu, Xiaoyun Yuan, and Lu Fang. Specat: Spatial-spectral cumulative-attention transformer for high-resolution hyperspectral image reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [[Paper]](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/CVPR2024/papers/Yao_SPECAT_SPatial-spEctral_Cumulative-Attention_Transformer_for_High-Resolution_Hyperspectral_Image_Reconstruction_CVPR_2024_paper.pdf) [[Code]](https://github.com/THU-luvision/SPECAT).
5. Yaohang Wu, Renwei Dian, and Shutao Li. Multistage spatial–spectral fusion network for spectral super-resolution. IEEE Transactions on Neural Networks and Learning Systems, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10713289), Code not Available.
6. Yihong Leng, Jiaojiao Li, Rui Song, Yunsong Li, and Qian Du. Uncertainty-guided discriminative priors mining for flexible unsupervised spectral reconstruction. IEEE Transactions on Neural Networks and Learning Systems, 2025. [[Paper]](https://ieeexplore.ieee.org/document/10843147) [[Code]](https://github.com/SuperiorLeo/Uncertainty-guided-UnSSR?tab=readme-ov-file).
7. Leng, Y., Li, J., Xu, H., & Song, R. (2025, March 10). From image- to pixel-level: Label‑efficient hyperspectral image reconstruction. arXiv preprint arXiv:2503.06852. [[Paper]](https://arxiv.org/abs/2503.06852), Code not available.
8. Feng, F., Cong, R., Wei, S., Zhang, Y., Li, J., Kwong, S., & Zhang, W. (2025, January 2). Unleashing correlation and continuity for hyperspectral reconstruction from RGB images. arXiv preprint arXiv:2501.01481. [[Paper]](https://www.arxiv.org/abs/2501.01481), Code not Available.

Diffusion Based models:
1. Shen, S., Pan, B., Zhang, Z., & Shi, Z. (2025, June 3). Hyperspectral image generation with unmixing guided diffusion model. arXiv preprint arXiv:2506.02601. [[Paper]](https://arxiv.org/abs/2506.02601), Code not Available.
2. Pang, L., Cao, X., Tang, D., Xu, S., Bai, X., Zhou, F., & Meng, D. (2024, September 19). HSIGene: A foundation model for hyperspectral image generation. arXiv preprint arXiv:2409.12470. [[Paper]](https://arxiv.org/html/2409.12470v2), Code not available. 

## Work Package 4 (Plant Disease Detection using HyperSpectral Images):
### Papers using HSI but not description:
1. Kuswidiyanto, L. W., Noh, H.-H., & Han, X. (2022). Plant Disease Diagnosis Using Deep Learning Based on Aerial Hyperspectral Images: A Review. Remote Sensing, 14(23), 6031. [[Paper]](https://www.mdpi.com/2072-4292/14/23/6031), Code not available.
2. García‑Vera, Y. E., Polochè‑Arango, A., Mendivelso‑Fajardo, C. A., & Gutiérrez‑Bernal, F. J. (2024). Hyperspectral Image Analysis and Machine Learning Techniques for Crop Disease Detection and Identification: A Review. Sustainability, 16(14), 6064. [[Paper]](https://www.mdpi.com/2071-1050/16/14/6064?utm_source=chatgpt.com), Code not available.
3. Lin, Y.-F., Cheng, C.-H., Qiu, B.-C., Kang, C.-J., Lee, C.-M., & Hsu, C.-C. (2024, August 31). Self‑supervised Fusarium Head Blight Detection with Hyperspectral Image and Feature Mining. arXiv preprint arXiv:2409.00395. [[Paper]](https://arxiv.org/abs/2409.00395).
4. Rahman, S. T., Vasker, N., Ahammed, A. K., & Hasan, M. (2024, September 18). Advancing Cucumber Disease Detection in Agriculture through Machine Vision and Drone Technology. arXiv preprint arXiv:2409.12350. [[Paper]](https://arxiv.org/abs/2409.12350), Code not available.
5. Zhang, K., Shi, Q. H., et al. (2025, July 30). Early detection of tomato leaf spot and wilt diseases based on hyperspectral imaging technology. Vegetable Research, 5, e026. [[Paper]](https://www.maxapress.com/data/article/vegres/preview/pdf/vegres-0025-0010.pdf), Code not available.
6. Pan, J., Lin, J., & Xie, T. (2024). Early detection of pine wilt disease based on UAV reconstructed hyperspectral image. Frontiers in Plant Science, Article 1453761. [[Paper]](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1453761/full), Code not available.

### Papers providing Multi-Modal functionalities (contains description but does not use HSI):
1. Li, J., Zhao, F., Zhao, H., Zhou, G., Xu, J., Gao, M., Li, X., Dai, W., Zhou, H., Hu, Y., & He, M. (2024, July 2). A multi‑modal open object detection model for tomato leaf diseases with strong generalization performance using PDC‑VLD. Plant Phenomics. [[Paper]](https://spj.science.org/doi/epdf/10.34133/plantphenomics.0220).
2. Dong, J., Yao, Y., Fuentes, Á., Jeong, Y., Yoon, S., & Park, D. S. (2024, December 1). Visual information guided multi‑modal model for plant disease anomaly detection. Smart Agricultural Technology.[[Paper]](https://www.sciencedirect.com/science/article/pii/S2772375524001734).
3. Pranith, P., Yeshwanth, V., & Thenmozhi, D. (2025, July 1). Multimodal few‑shot learning for plant disease detection with contrastive pre‑training and query addressal. Neural Computing and Applications. [[Paper]](https://link.springer.com/article/10.1007/s00521-025-11438-5).
4. Kolluri, J. ., Dash, S. K. ., & Das, R. . (2024). Plant Disease Identification Based on Multimodal Learning. International Journal of Intelligent Systems and Applications in Engineering, 12(15s), 634–643. [[Paper]](https://www.ijisae.org/index.php/IJISAE/article/view/4815).
5. KİLİM, Oğuzhan and Yiğit, Tuncay and ARMAĞAN, Hamit, Cvxlm: An Explainable Multi-Modal Deep Learning Framework for Plant Disease Diagnosis with Natural Language Reporting. [[Paper]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5370311).
6. Roumeliotis, K. I., Sapkota, R., Karkee, M., Tselikas, N. D., & Nasiopoulos, D. K. (2025). Plant Disease Detection through Multimodal Large Language Models and Convolutional Neural Networks. arXiv preprint arXiv:2504.20419. [[Paper]](https://arxiv.org/abs/2504.20419).
7. Wei, T., Chen, Z., Huang, Z., & Yu, X. (2024, October). Benchmarking In‑the‑Wild Multimodal Plant Disease Recognition and A Versatile Baseline. In Proceedings of the 32nd ACM International Conference on Multimedia (MM ’24), Melbourne, VIC, Australia (pp. —). [[Paper]](https://arxiv.org/html/2408.03120v1) [[Code]](https://github.com/tqwei05/MVPDR).

### Papers using both HSI and Description:
None Found till now
