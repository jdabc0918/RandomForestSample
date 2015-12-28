#pragma once

#include "stdafx.h"

#define LABEL_NUM 10
#define PATCH_SIZE 8

class RandomForest
{
private:
    /**
    *   @struct 特徴ベクトル
    **/
    typedef struct FeatureVector
    {
        vector<Mat>  pv;   /**< 輝度パッチ*/
        vector<Mat>  edge; /**< エッジパッチ*/
    };

    /**
    *   @struct サンプルデータ情報
    **/
    typedef struct Data
    {
        int label;              /**< ラベル値*/
        FeatureVector feature;  /**< 特徴ベクトル*/
    };

    /**
    *   @struct 分岐関数パラメータ
    **/
    typedef struct SplitParam
    {
        vector<bool> enable;        /**< 使うかどうかのテーブル*/
        vector<int>  pv_th;      /**< 輝度閾値*/
        vector<int>  edge_th;    /**< エッジ閾値*/
    };

    /**
    *   @struct 決定木構造体
    **/
    typedef struct Tree
    {
        SplitParam param;   /**< 分岐関数パラメータ*/
        Tree *left;         /**< 子ノード(左)へのポインタ*/
        Tree *right;        /**< 子ノード(右)へのポインタ*/
        Tree *parent;       /**< 親ノードへのポインタ*/
        vector<double> Probs;   /**< 各ラベルの存在確率(葉だけ)*/
        Tree()
        {
            left = NULL;
            right = NULL;
            parent = NULL;
            Probs.assign(LABEL_NUM, 0.0);
        }
        ~Tree(){}
        /**
        *   自信が親ノードかどうかを返す
        *   @return true(親)/false(それ以外)
        **/
        bool isRoot(void)
        {
            return (parent == NULL) ? true : false;
        }
        /**
        *   自信が葉(末端ノード)かどうかを返す
        *   @return true(葉)/false(それ以外)
        **/
        bool isLeaf(void)
        {
            return (left == NULL && right == NULL) ? true : false;
        }
        /**
        *   深さを返す
        *   @return 深さ(ノードは0)
        **/
        int getDepth(void)
        {
            if (isRoot()) return 0;
            else return 1 + parent->getDepth();
        }
    };

private:
    vector<Tree> Forest;    /**< 学習フォレスト*/
    vector<Data> Samples;   /**< 学習サンプル*/

    /**
    *   特徴ベクトルを計算する
    *   @param [in] _in 画像データ
    *   @return 特徴ベクトル
    **/
    FeatureVector calcFeature(Mat _in);
    /**
    *   学習サンプルを読み込む
    **/
    void LoadSamples(void);
    /**
    *   一様乱数を発生させる
    *   @return 0.0~1.0の乱数
    **/
    double rand_uniform(void);
    /**
    *   重複のない乱数リストを生成する
    *   @param [in] _size 要素数
    *   @param [in] _min  下限
    *   @param [in] _max  上限
    **/
    vector<int> make_rand_array_unique(int _size, int _min, int _max);
    /**
    *   学習データをランダムに抽出する
    *   @param [in] _num 抽出数
    *   @return 抽出データ
    **/
    vector<Data> getRandomSampleSubset(int _num);
    /**
    *   パラメータをランダムに生成する
    *   @return 生成されたパラメータ
    **/
    SplitParam getRandomParameter(void);
    /**
    *   データのエントロピーを計算する
    *   @param [in] _data 計算対象データ
    *   @return エントロピー値
    **/
    double calcDataEntropy(vector<Data> _data);
    /**
    *   分岐関数
    *   @param [in] _in    入力データ
    *   @param [in] _param パラメータ
    *   @return true(左)/false(右)
    **/
    bool splitFunction(Data _in, SplitParam _param);
    /**
    *   葉のラベル存在確率を計算
    *   @param [in] _data データリスト
    *   @return ラベルの存在確率
    **/
    vector<double> calcProbs(vector<Data> _data);
    /**
    *   木を学習する
    *   @param [out] _tree 学習対象の木
    *   @param [in]  _data 学習に使うサンプルデータ
    **/
    void LearnTree(Tree *_tree, vector<Data> _data);
    /**
    *   木の識別結果を返す
    *   @param [in] _tree 対象木
    *   @param [in] _fv 入力特徴ベクトル
    *   @param [out] _res 識別結果
    **/
    void TestTree(Tree *_tree,FeatureVector _fv, vector<double> &_res);
public:
    RandomForest();     /**< コンストラクタ*/
    ~RandomForest();    /**< デストラクタ*/
    /**
    *   森を学習する
    **/
    void LearnForest();
    /**
    *   識別する
    *   @param [in] _img 入力画像
    *   @param [out] _label 識別結果
    **/
    void TestForest(Mat _img,int &_label);
};