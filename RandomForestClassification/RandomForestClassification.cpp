// RandomForestClassification.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"


int main(int argc, char **argv)
{
    RandomForest rf;
    //学習
    rf.LearnForest();
    //テスト
    int labelNum = 10;
    int sampleNum = 1000;
    string imgName = "C:/Users/Yosuke/Desktop/CIFAR10/test/%d/%04d.bmp";
    for (int i = 0; i < labelNum; i++)
    {
        int counter = 0;    //正しく識別できたら++
        for (int j = 0; j < sampleNum; j++)
        {
            char fname[FILENAME_MAX];
            sprintf(fname, imgName.c_str(), i, j);
            Mat img = imread(fname, 0);

            int label;
            rf.TestForest(img, label);

            if (label == i) counter++;
        }

        //結果の出力
        cout << "識別率[" << i << "]:" << (double)counter / sampleNum * 100 << "[%]" << endl;
    }
	return 0;
}

