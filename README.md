# Heart-Disease-Prediction-ML
# Projenin Amacı:
Bu projede Kaggle'da yer alan "Indicators of Heart Disease" veri seti kullanılarak hastaların sahip olduğu çeşitli faktörlere göre kalp hastalığı olup olmadığının tahmin edilmesi amaçlanmıştır.

Projenin Kaggle notebook linki : https://www.kaggle.com/code/aycaen/heart-disease-prediction-ml/notebook

# Veri Seti:
Kullanılan veri seti 319795 satırdan ve 18 sütundan oluşmaktadır.

- HeartDisease : Kişilerin kalp hastalığının olup olmadığını gösterir.
- BMI : Kişilerin beden kitle indeksini gösterir.
- Smoking : Kişilerin hayatı boyunca en az 100 sigara içip içmediğini gösterir.
- AlcoholDrinking : Ağır alkol içicileri gösterir.
- Stroke : Kişilerin daha önce felç geçirip geçirmediklerini gösterir.
- PhysicalHealth : Kişilerin son 30 gün boyunca kaç gün fiziksel sağlıklarının bozuk olduğunu gösterir.
- MentalHealth : Kişilerin son 30 gün boyunca kaç gün mental sağlıklarının bozuk olduğunu gösterir.
- DiffWalking : Kişilerin yürümede ya da merdivenleri çıkmada ciddi bir zorluk geçirip geçirmediklerini gösterir.
- Sex : Cinsiyetleri gösterir.
- AgeCategory : 14 ayrı yaş kategorilerini gösterir.
- Race : Etnik kökenleri, ırkları gösterir.
- Diabetic : Kişilere diyabet tanısı konup konmadığını gösterir.
- PhysicalActivity : Kişilerin son 30 gün boyunca günlük işlerinin dışında fiziksel bir aktivitede bulunup bulunmadığını gösterir.
- GenHealth : Kişilerin genel sağlıklarının nasıl olduğunu gösterir.
- SleepTime : Kişilerin 24 saatte ortalama kaç saat uyuduklarını gösterir.
- Asthma : Kişilere daha önce astım tanısı konup konmadığını gösterir.
- KidneyDisease	: Kişilere böbrek taşları, mesane enfeksiyonu veya idrar kaçırma hariç hiç böbrek hastalığı teşhisi konup konmadığını gösterir.
- SkinCancer : Kişilere hiç cilt kanseri teşhisi konup konmadığını gösterir.

## Veri Seti için Kullanılan Kütüphaneler ve Algoritmalar
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_validate
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import adjusted_rand_score, v_measure_score
```
# Projenin İçeriği ve Kullanılan Yöntemler
Bu projede ilk önce keşifsel veri analizi yapılarak veri seti hakkında genel bilgiler edinilmiş, kategorik ve nümerik değişkenler görselleştirilmiş, kategorik ve nümerik değişkenlerin kalp hastalığı üzerindeki etkisi grafikler üzerinden gösterilmiştir. Veri setinin dengesiz bir veri seti olduğu, kalp hastalığına sahip kişilerin gözlem sayısının çok düşük olduğu tespit edilmiştir. Daha sonra veri ön işleme adımı yapılarak tekrar eden satırların temizlenmesi, aykırı değerlerin bulunması, kategorik değişkenler için label encoding işlemleri yapılmıştır. 
Veri setinin bağımsız ve bağımlı değişkenleri belirlenerek test ve train şeklinde ayrılarak lojistik regresyon gerçekleştirilmiş ve çıkan sonuçta 0 sınıfına ait tahminlerin çok iyi olduğu ama 1 sınıfına ait sonuçların çok kötü olduğu görülmüştür.
Bu sorunu çözmek için over-sampling yöntemi olan SMOTE işlemi ile azınlıkta olan gözlem sayısı arttırılarak dengesiz veri seti daha dengeli bir hale getirilmiştir. Tekrar lojistik regresyon uygulanarak önceki sonuçlar ile karşılaştırılmıştır.
Gözetimsiz öğrenme tekniklerinden olan KMeans algoritması kullanılmıştır. Öncelikle elbow yani dirsek yöntemi ile en uygun küme sayısı belirlenmiş daha sonra bu küme sayısını kullanılarak KMeans kümeleme yöntemi gerçekleştirilmiştir.
Grafiğe dökülerek yorumlamalar yapılmıştır.

# Proje Sonucu
Proje sonucunda KMeans kümeleme yönteminin iyi bir sonuç vermediği gözlenmiştir. Lojistik regresyonun ise veri setini dengeli bir hale getirdikten sonra daha iyi bir sonuç verdiği görülmüştür.
