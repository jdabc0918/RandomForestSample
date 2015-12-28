// RandomForestClassification.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"


int main(int argc, char **argv)
{
    RandomForest rf;
    //�w�K
    rf.LearnForest();
    //�e�X�g
    int labelNum = 10;
    int sampleNum = 1000;
    string imgName = "C:/Users/Yosuke/Desktop/CIFAR10/test/%d/%04d.bmp";
    for (int i = 0; i < labelNum; i++)
    {
        int counter = 0;    //���������ʂł�����++
        for (int j = 0; j < sampleNum; j++)
        {
            char fname[FILENAME_MAX];
            sprintf(fname, imgName.c_str(), i, j);
            Mat img = imread(fname, 0);

            int label;
            rf.TestForest(img, label);

            if (label == i) counter++;
        }

        //���ʂ̏o��
        cout << "���ʗ�[" << i << "]:" << (double)counter / sampleNum * 100 << "[%]" << endl;
    }
	return 0;
}

