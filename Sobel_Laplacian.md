### Оператор Собеля ###

***

#### Введение ####
Одна из важнейших [сверток](https://github.com/TryOpenCV/sobel/blob/master/Plexus.md) – это вычисление производных. 
В математике и физике производные играют очень важную роль, то же самое можно сказать и про компьютерное зрение.
Изображения, с которыми мы работаем, состоят из пикселей, которые, для картинки в градациях серого, задают значение яркости.
Т.е. наша картинка — это просто двумерная матрица чисел. Поэтому производная в облости работы с изображениями — это отношение значения приращения пикселя по y к значению приращению пикселя по x.
```
dI = dy/dx
```
Работая с изображением **I**, мы работает с функцией двух переменных **I(x,y)**, т.е. со скалярным полем. Поэтому, более правильно говорить не о производной, а о градиенте изображения.
>**Градиент** (от лат. gradiens — шагающий, растущий) — вектор, показывающий направление наискорейшего возрастания некоторой величины, значение которой меняется от одной точки пространства к другой
(скалярного поля).

Итак, градиент для каждой точки изображения (функция яркости) — двумерный вектор, компонентами которого являются производные яркости изображения по горизонтали и вертикали. 
```
grad I(x,y) = (dI/dx, dI/dy)
```

###### Упрощенное описание ######
Оператор вычисляет градиент яркости изображения в каждой точке. Так находится направление наибольшего увеличения яркости и величина её изменения в этом направлении. Результат показывает, насколько «резко» или «плавно» меняется яркость изображения в каждой точке, а значит, вероятность нахождения точки на грани, а также ориентацию границы. На практике, вычисление величины изменения яркости (вероятности принадлежности к грани) надежнее и проще в интерпретации, чем расчёт направления.

#### Теория ####

Оператор Собеля используется в области обработки изображений. Часто его применяют в алгоритмах выделения границ. По сути, это дискретный дифференциальный оператор, вычисляющий приближенное значение градиента яркости изображения. Результатом применения оператора Собеля в каждой точке изображения является либо вектор градиента яркости в этой точке, либо его норма. Оператор Собеля основан на свёртке изображения небольшими сепарабельными целочисленными фильтрами в вертикальном и горизонтальном направлениях, поэтому его относительно легко вычислять.

Математически, градиент функции двух переменных для каждой точки изображения (которой и является функция яркости) — двумерный вектор, компонентами которого являются производные яркости изображения по горизонтали и вертикали. В каждой точке изображения градиентный вектор ориентирован в направлении наибольшего увеличения яркости, а его длина соответствует величине изменения яркости. Это означает, что результатом оператора Собеля в точке области постоянной яркости будет нулевой вектор, а в точке, лежащей на границе областей различной яркости — вектор, пересекающий границу в направлении увеличения яркости.

Рассмотрим пример:

Пусть мы хотим найти некоторые границы на изображении.

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_0.jpg)

Высокое изменение градиента указывает на значительные изменения в изображении.
Для более наглядного примера посчитаем, что у нас есть 1D-изображение.

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg)

Более отчетливо можно увидет и определить изменения, если мы возьмем первую производную:

![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg)

Таким образом можно сделать вывод, что для определения границы изображения нужно выбирать тот пиксель, у которого наиболее высокий градиент по сравнению с соседними.

#### Формализация ####
Оператор собеля использует ядра 3×3, с которыми сворачивают исходное изображение для вычисления приближенных значений производных по горизонтали и по вертикали. Пусть A исходное изображение, а Gx и Gy — два изображения, где каждая точка содержит приближенные производные по x и по y. Они вычисляются следующим образом:

![](https://upload.wikimedia.org/math/9/8/a/98a9514f83f3e81531216e92efb212a3.png)
(* обозначает двумерную операцию свертки.)

В каждой точке изображения приближенное значение величины градиента можно вычислить, используя полученные приближенные значения производных:

![](https://upload.wikimedia.org/math/4/9/4/4941373cd3ab2fe1b807a06e879c323b.png)

Иногда для простоты используется:

![](http://docs.opencv.org/_images/math/a938de1258efc69f677cc4ff9d6c430b3530580a.png)

#### Алгоритм ####
1. Загружаем изображение
2. Применяем размытие по Гауссу (для уменьшения шума).
3. Конвертируем изображение в оттенки серого.
4. Используя оператор Собеля получаем градиент по X.
5. Используя оператор Собеля получаем градиент по Y.
6. Аппроксимируем градиенты по X и по Y
7. Выводим новое изображение.  

Примечание:
>тобы избежать переполнения целевое изображение должно быть 16-битным (IPL_DEPTH_16S) при 8-битном исходном изображении  

#### Пример кода ####
Применяет оператор Собеля и выдает на выходе изображение с обнаруженными краями (яркие пиксели на более темном фоне).
```c++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{

  Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;

  /// Загружаем изображение
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  /// Применяем размытие по Гауссу
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Конвертируем его
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// Создаем новое окно
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Объявляем градиенты grad_x и grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Градиент по X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x ); // модуль значения градиента 

  /// Градиент по Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y ); // модуль значения градиента 

  /// Слияние градиентов (аппроксимация)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( window_name, grad );

  waitKey(0);

  return 0;
  }
```
#### Поясниния ####
Объявляем переменные:
```c++
Mat src, src_gray;
Mat grad;
char* window_name = "Sobel Demo - Simple Edge Detector";
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
```
Загружаем изображение:
```c++
src = imread( argv[1] );

if( !src.data )
{ return -1; }
```
Сначала применяем функцию [GaussianBlur](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur) к нашему изображению для уменьшения шума (размер ядра 3x3):
```c++
GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
```
Конвертируем изображение в формат "оттенки серого":
```c++
cvtColor( src, src_gray, CV_RGB2GRAY );
```
Вычисляем производные по горизонтальному и вертикальному направлению (по х и у). Для этого используем функцию Собеля:
```c++
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;

/// Градиент по X
Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
/// Градиент по Y
Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
```
```
C++: void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
```
**Аргументы функции:**
* **src** - входное изображение.
* **dst** - изображение на выходе (такой же разимер и кол-во каналов как и у входного изображения).
* **ddepth** - [глубина цвета](https://ru.wikipedia.org/wiki/%D0%93%D0%BB%D1%83%D0%B1%D0%B8%D0%BD%D0%B0_%D1%86%D0%B2%D0%B5%D1%82%D0%B0)  (изображения). Может быть определена как src.depth():
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
* **dx** - пороядок производной по X
* **dy** - пороядок производной по Y
* **ksize** - размер ядра свертки (может быть равным: 1, 3, 5 или 7)
* **scale** - опционально масштабный коэффициент для вычисленных производных величин; по умолчанию, масштабирование не применяется (см [getDerivKernels ()](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#void getDerivKernels(OutputArray kx, OutputArray ky, int dx, int dy, int ksize, bool normalize, int ktype)) для более подробной информации).
* **delta** - необязательно дельта значение, которое добавляется к каждому элементу результата перед сохранением его в  **dst**.
* **borderType** - задает метод экстрополяции пикселей на изображении (см [borderInterpolate()](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#int borderInterpolate(int p, int len, int borderType)) для более подробной информации).

Конвертируем наш промежуточный результат обратно в CV_8U (т.к. при вызове функции Sobel мы указали CV_16S формат, чтобы избежать переполнения):
```c++
convertScaleAbs( grad_x, abs_grad_x );
convertScaleAbs( grad_y, abs_grad_y );
```
Аппроксимируем изображение:
```c++
addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
```
```
C++: void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1)
```
Резултат: 
![](http://docs.opencv.org/_images/math/160c3479896ac799bb5c7d260a052e6b35c463ef.png)

Выводим изображение:
```c++
imshow( window_name, grad );
```
#### Результат ####
![](http://docs.opencv.org/_images/Sobel_Derivatives_Tutorial_Result.jpg)


### Оператор Лапласа ###

***

#### Введение ####
После рассмотрения оператора Собеля, можно перейти к оператору Лапласа, который позволяет вычислить т.н. лапласиан изображения — суммирование производных второго порядка.

#### Теория ####
Используя первую производную интенсивности (оператор Собеля), мы смогли найти граничный пиксель (выделить некатый контур). Это видно по "скачку" фукции произдводной:
![](http://docs.opencv.org/_images/Laplace_Operator_Tutorial_Theory_Previous.jpg)

Но что будет, если мы возьмем вторую производную? Рассмотрим график второй производной:

![](http://docs.opencv.org/_images/Laplace_Operator_Tutorial_Theory_ddIntensity.jpg)

Можно заметить, что вторая производная , в точке максимума первой, равна нулю. Таким образом, мы можем также использовать этот критерий, чтобы попытаться обнаружить контуры в изображении. Тем не менее, значение производной может обращаться в нуль и в других точках. Эта проблема может быть решена путем предварительной фильтрации изображения.
Из приведенного выше объяснения, получаем, что вторая производная может быть использована для обнаружения контура на изображении. Когда мы говорим про изображение, мы рассматриваем "2D" пространство. Поэтому оператор Лапласа принимает вид:

![](http://docs.opencv.org/_images/math/b7e0e54736500f2886c2fa2118852f1fa01d238e.png)

Оператор Лапласа реализован в OpenCV как функция [Laplacian](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=laplacian#laplacian)
Т.к функция Laplacian работает с градиентом изображения, то основой для ее реализации является оператор Собеля (функция [Sobel](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=laplacian#void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType))).
#### Пример кода ####
Данный пример кода:
* загружает зображение;
* устраняет шум путем применения размытия по Гауссу;
* преобразовывает исходное изображение в оттенки серого;
* применяет оператор Лапласа (на выходе черно-белое изображение);
* выводит и сохраняет полученный результат.
```c++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{
  Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  char* window_name = "Laplace Demo";

  int c;

  /// Загружаем изображение
  src = imread( argv[1] );

  if( !src.data )
    { return -1; }

  /// Убираем шум, используя гауссовский фильтр
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// конвертируем изображение в оттенки серого
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// создаем новое рабочее окно
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Применяем функцию Laplacian (оператор Лапласа)
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );

  /// выводим результат
  imshow( window_name, abs_dst );

  waitKey(0);

  return 0;
  }
```

#### Пояcнения ####
Инициализируем необходимые переменные:
```c++
Mat src, src_gray, dst;
int kernel_size = 3;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
char* window_name = "Laplace Demo";
```
Загружаем изображение:
```c++
src = imread( argv[1] );

if( !src.data )
  { return -1; }
```
Применяем гауссово размывание для того что бы уменьшить шумы:
```c++
GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
```
Конвертируем изображение в оттенки серого, исрользуя функцию [cvtColor](http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor):
```c++
cvtColor( src, src_gray, CV_RGB2GRAY );
```
Применяем оператор Лапласа:
```c++
Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
```
#####Функция Laplacian#####
```
C++: void Laplacian(InputArray src, OutputArray dst, int ddepth, int ksize=1, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
```
Аргументы функции:
* **src_gray** - исходное изображение в оттенках серого;
* **dst** - изображение для сохранения результа;
* **ddepth** - [глубина цвета](https://ru.wikipedia.org/wiki/%D0%93%D0%BB%D1%83%D0%B1%D0%B8%D0%BD%D0%B0_%D1%86%D0%B2%D0%B5%D1%82%D0%B0)  (изображения). Может быть определена как src.depth():
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
    * src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
* **ksize** - размер апертуры, используются для вычисления второй производной (см. [getDerivKernels()](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=laplacian#int borderInterpolate(int p, int len, int borderType))).Значение переменной должено быть положительным и нечетным. По умолчанию принимает значение равное 1;
* **delta** - необязательно дельта значение, которое добавляется к каждому элементу результата перед сохранением его в dst;
* **borderType** - задает метод экстрополяции пикселей на изображении (см. borderInterpolate() для более подробной информации).
Вычисляет значение второй производной по X и Y используя оператор Собеля:

![](http://docs.opencv.org/_images/math/b1873bdbd97ad13406fd470230dafc8b00683d8b.png)

Преобразуем полученное изображение в формат CV_8U и выводим на экран:
```
convertScaleAbs( dst, abs_dst );
imshow( window_name, abs_dst );
```
#### Результат ####
Исходное изображение:

![](http://docs.opencv.org/_images/Laplace_Operator_Tutorial_Original_Image.jpg)

Результат. Обратите внимание на четкие контуры деревьев и коровы (за исключением облостей, где интенсивности соседних пикселей схожи - вокруг головы коровы):

![](http://docs.opencv.org/_images/Laplace_Operator_Tutorial_Result.jpg)