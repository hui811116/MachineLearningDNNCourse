#ifndef UTIL_H
#define UTIL_H
#include <random>
#include <ctime>
#include <iostream>
#include <cassert>

using namespace std;

class NnGen{
	public:
		NnGen(){
			_dis = new normal_distribution<float>(0,1);
			_m=0;_v=1;
		}
		NnGen(float mean,float var){
			assert(var>0);
			_m=mean;_v=var;
			_dis = new normal_distribution<float>(mean,var);
		}
		~NnGen(){delete _dis;}
		void reset(float mean, float var){
			assert(var>0);
			delete _dis;
			_dis=new normal_distribution<float>(mean,var);
		}
		float operator ()(){
			return _dis->operator()(_gen);
		}
		void showParam(){
			cout<<"mean: "<<_m<<" var: "<<_v<<endl;
		}
	private:
		float _m;
		float _v;
		static mt19937 _gen;
		normal_distribution<float>* _dis;

};
mt19937 NnGen::_gen((unsigned)time(NULL));
extern NnGen gn(0,0.2);
#endif


