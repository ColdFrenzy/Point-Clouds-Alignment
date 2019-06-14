#include "myheader.h"
#define MAXBUFSIZE  ((int) 1e6)
using namespace std;
int DIM = 3;
int N_RAND = 3;     //number of random elements
Eigen::VectorXd leaf_range = Eigen::Vector3d(0.001,0.04,0.06);
Eigen::VectorXd query_distance = Eigen::Vector3d(0.01,2.0,1);
double point_dist = 0.4;
int trial = 10000;  //10000
int iterations = 300; 
Eigen::VectorXd error_vec = Eigen::VectorXd::Zero(DIM); 
bool notfound = false;
int n_err = 0;
bool train_ransac = false;
bool train_icp = false;
bool show_ransac = true;
bool show_icp = true; 
bool my_test = false;

        
int main(int argc, char** argv){

    for(int i = 0; i < DIM; i++){
        error_vec(i) = -1;
    }
    //take the points of the scene and draw it 
    string map_file("./08-find_that_object/globe-in-scene.txt");
    string object_file("./08-find_that_object/globe.txt");
    string out_icp("./out_icp.txt");
    string out_ransac("./out_ransac.txt");
    VectorXdVector object_points = get_points(object_file,DIM);
    VectorXdVector map_points = get_points(map_file,DIM);
    if(DIM <= 3){
        draw_distance_map(map_points,object_points);
    }
    transf bestT;
    transf FinalT;
    transf initialT;
    initialT.R = Eigen::Matrix3d::Zero();
    for(int j = 0; j<DIM; j++){
        initialT.R(j,j) = 1.0;
    }
    initialT.t = Eigen::Vector3d(0.0,0.0,0.0);
    int bestInlier = 0;
    //initialize random seed
    srand(time(NULL));


    BaseTreeNode* map_tree=buildTree(map_points, leaf_range(0)); 
    if(train_ransac == true){
        cout << "***************************************************************"<<endl;
        cout << "*************************RANSAC STARTS*************************"<< endl;
        cout << "***************************************************************"<<endl;
        for(int k = 0;k < leaf_range.rows()-2; k++){


            // build the kdtree of the map points
            BaseTreeNode* map_tree=buildTree(map_points, leaf_range(k));
            cout << "Actual leaf range: " << leaf_range(k)<< endl;
            int initialInlier = getScore(map_points,object_points,initialT,map_tree,
                    query_distance(k));
            if(initialInlier > bestInlier){
                cout << "starting inlier number: "<< initialInlier<<endl;
                bestInlier = initialInlier;
                bestT = initialT;
                //system("read -p 'Press Enter to continue...' var");
            }
            for(int i = 0; i < trial; i++ ){
                printf("picking %d random points\n", N_RAND);
                int map_element = map_points.size();
                n_err = 0;
                notfound = false;
                VectorXdVector random_points = enhancedRandomPick(map_points,
                        object_points, N_RAND,point_dist);
                VectorXdVector rand_map(N_RAND);
                VectorXdVector rand_obj(N_RAND);
                for(int j = 0; j< N_RAND; j++){
                    rand_obj[j] = random_points[j];
                    rand_map[j] = random_points[j+N_RAND];
                }
                for(int j = 0; j< 2*N_RAND; j++){
                    if (random_points[j] == error_vec ){
                        n_err += 1;
                    }
                }
                if(n_err == 2*N_RAND){
                    notfound = true;
                }

                if(!notfound){
                    transf T = ICPLinearRelaxation(rand_map,rand_obj);


                    int num_inliers = getScore(map_points,object_points,T,map_tree,
                            query_distance(k));
                    if (num_inliers > bestInlier){
                        bestT = T; 
                        bestInlier = num_inliers;
                    }
                    cout <<"iteration "<< i <<" number of inliers: " << num_inliers <<endl;
                    cout << "best number of inliers: "<< bestInlier<<endl;
                }
            }
        }

        VectorXdVector initial_guess(object_points.size());
        for(int i = 0; i < object_points.size(); i++){
            initial_guess[i] = bestT.R*object_points[i] + bestT.t; 
        }
        draw_distance_map(map_points,initial_guess);    
        ofstream outfile1;
        outfile1.open(out_ransac);
        outfile1<<bestT.R<<"\n";

        outfile1<<bestT.t.transpose();
        outfile1.close(); 
    }
    if(train_icp == true){
        cout << "************************************************************"<<endl;
        cout << "*************************ICP STARTS*************************"<<endl;
        cout << "************************************************************"<<endl;
        if(train_ransac == false){
            bestT = read_transf_file(out_ransac,MAXBUFSIZE);
        }
    
        Eigen::VectorXd VectorT(6);
        VectorT = ICP(bestT, object_points,map_tree,query_distance(1),iterations,1e-3);
        FinalT = v2t(VectorT);
        VectorXdVector new_obj(object_points.size());
        for(int i = 0; i < object_points.size(); i++){
            new_obj[i] = FinalT.R*object_points[i] + FinalT.t; 
        }
        // let's write on file
        ofstream outfile;
        outfile.open(out_icp);
        outfile<<FinalT.R<<"\n";

        outfile<<FinalT.t.transpose();
        outfile.close();    
        draw_distance_map(map_points,new_obj);
    }
    if(show_ransac){
        if(train_ransac == false){
            bestT = read_transf_file(out_ransac,MAXBUFSIZE);
        }
        VectorXdVector initial_guess(object_points.size());
        cout << "R" << bestT.R << endl << "t" << bestT.t <<endl;
        for(int i = 0; i < object_points.size(); i++){
            initial_guess[i] = bestT.R*object_points[i] + bestT.t; 
        }
        draw_distance_map(map_points,initial_guess);
    }
    if(show_icp){
        if(train_icp == false){
            FinalT = read_transf_file(out_icp,MAXBUFSIZE);
        }
        VectorXdVector new_obj(object_points.size());
        for(int i = 0; i < object_points.size(); i++){
            new_obj[i] = FinalT.R*object_points[i] + FinalT.t; 
        }
        draw_distance_map(map_points,new_obj);
    } 
    if(my_test){
        if(train_ransac == false){
            bestT = read_transf_file(out_ransac,MAXBUFSIZE);
        }
        VectorXdVector initial_guess(object_points.size());
        Eigen::VectorXd my_x(3);
        my_x = t2v(bestT);
        Eigen::VectorXd vec2(6);
        vec2 << bestT.t(0), bestT.t(1), bestT.t(2), my_x(0),my_x(1),my_x(2);
        transf my_transf = v2t(vec2);
        for(int i = 0; i < object_points.size(); i++){
            initial_guess[i] = my_transf.R*object_points[i] + my_transf.t; 
        }
        draw_distance_map(map_points,initial_guess);
    }
    return 0;
}



