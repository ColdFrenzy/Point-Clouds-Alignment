#include "myheader.h"
using namespace std;
using namespace cv;


VectorXdVector get_points(string file, int dim){
    string line;
    ifstream myfile;
    double val;
    int n_elem = 0;
    vector<double> all_points;
    myfile.open(file);
    while( getline (myfile,line) ){
        n_elem ++;
        istringstream iss(line);
        while(iss >> val){
            all_points.insert(all_points.end(),val);
        }
    }
    // now that we have all the points inside the vector we create our EigenVec
    VectorXdVector point_map(n_elem);
    for (size_t i=0; i<n_elem; i++){
        Eigen::VectorXd point(dim);
        for (int k=0; k<dim; k++){
            point(k)= all_points[i*dim + k];
        }
        point_map[i]=point;
    }
    return point_map;
}

// return a vector with 2N points
VectorXdVector randomPick(const VectorXdVector& map,
        const VectorXdVector& obj, int N){
    
    int map_elem = map.size();
    int obj_elem = obj.size();
    
    VectorXdVector points(N*2);
    int index = 0;
    
    for(int i = 0; i < N; i++){
        index = rand() %map_elem;
        points[i] = map[index];
    }
    
    for(int i = N; i < 2*N; i++){
        index = rand() %obj_elem;
        points[i] = obj[index];
    }
    return points;
}

transf ICPLinearRelaxation(const VectorXdVector& map,
        const VectorXdVector& obj){
    int n_elem = map.size();
    double inv_nelem = 1.0d/n_elem; 
    int dim = map[0].rows();
    Eigen::VectorXd map_mean(dim);
    Eigen::VectorXd obj_mean(dim);
    VectorXdVector map_no_mean(n_elem);
    VectorXdVector obj_no_mean(n_elem);
    // compute the means
    for(int i = 0; i < n_elem; i++){
        map_mean += map[i];
        obj_mean += obj[i];
    }
    map_mean = map_mean * inv_nelem;
    obj_mean = obj_mean * inv_nelem;
    
    //subtract the mean 
    for(int i = 0; i < n_elem; i++){
        map_no_mean[i] = map[i] - map_mean;
        obj_no_mean[i] = obj[i] - obj_mean;
    }
    
    // compute the matrix
    double val = 0.0;
    Eigen::MatrixXd A(dim,dim);
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            for(int k = 0; k < dim; k++){
                val += map_no_mean[k](i)*obj_no_mean[k](j); 
//                val += obj_no_mean[k](i)*map_no_mean[k](j); 
            }
            A(i,j) =  val;
            val = 0.0;
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    int m = U.rows();
    int n = V.rows();
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m,n);
    Eigen::VectorXd S_values = svd.singularValues();
    int minval = 0;
    if(m >= n){
        minval = n;
    }else{
        minval = m;
    }
    for(int i = 0; i < minval; i++){
        S(i,i) = S_values(i);
    }
    Eigen::MatrixXd R = U * V.transpose();
    Eigen::VectorXd t(dim);
    t = map_mean - R*obj_mean;
    transf T;
    T.R = R;
    T.t = t;
    return T;
}



int getScore(const VectorXdVector& map, const VectorXdVector& obj,
        const transf& T, BaseTreeNode* map_tree,const double max_distance){
    // Transform all the object points
    int obj_size = obj.size();
    int num_inliers = 0;
    VectorXdVector new_obj(obj_size);
    for(int i = 0; i < obj_size; i++){
        new_obj[i] = T.R*obj[i] + T.t; 
    }
    
    
    // check the number of inliers
    for(int i = 0; i < obj_size; i++){
        Eigen::VectorXd query_point=new_obj[i];
        Eigen::VectorXd answer;
	double bf_distance=map_tree->findNeighbor(answer, query_point, max_distance);
	if (bf_distance>=0){
            num_inliers += 1;
        }
    }
    
    return num_inliers;
    }
    
   
    

VectorXdVector enhancedRandomPick(const VectorXdVector& map,const VectorXdVector& obj, 
            int n_elem, double dist){
    
    bool notfound1 = true;
    bool notfound2 = true;
    bool notfound = true;
    int dim = map[0].rows();
    int counter = 0;
    int max_count = 1000000;

    int N_map = map.size();
    int N_obj = obj.size(); 
    int obj_ind;
    int map_ind;
    VectorXdVector points_from_obj(n_elem);
    VectorXdVector points_from_map(n_elem);
    Eigen::VectorXd error_vec = Eigen::VectorXd::Zero(dim);
    for(int i = 0; i < dim; i++){
        error_vec(i) = -1;
    }
    
    
    while(counter < max_count && notfound){
        if(!notfound1 && !notfound2){
            notfound = false;
        }
        // take 3 numbers from the obj vector
       /* if (counter%10000 == 0){
            cout << "iteration number "<<counter<<endl;
        }*/
        if(notfound1){
            for(int i = 0; i < n_elem; i++){
                obj_ind = rand()%N_obj;
                points_from_obj[i] = obj[obj_ind];
            }
            if(checkDist(points_from_obj,n_elem,dist)){
                cout << "Object points founded"<<endl;
                notfound1 = false;
            }
        }
        // take 3 numbers from the map vector
        if(notfound2){
            for(int j = 0; j < n_elem; j++){
                map_ind = rand()%N_map;
                points_from_map[j] = map[map_ind];
            }
            if(checkDist(points_from_map,n_elem,dist)){
                cout << "Map points founded"<<endl;
                notfound2 = false;
            }
        }
        counter += 1;
    
    }
    if(counter == max_count){
        cout <<n_elem <<" random points at distance " << dist << " was not found"<<endl;
        VectorXdVector notf_vec(2*n_elem);
        for(int i = 0; i < 2*n_elem; i++){
            notf_vec[i] = error_vec;
        }
        return notf_vec;
    }else{
        cout << "Points founded"<<endl;
        VectorXdVector f_vec(2*n_elem);
        for(int i = 0; i < n_elem; i++){
            f_vec[i] = points_from_obj[i];
        }        
        for(int i = n_elem; i < 2*n_elem; i++){
            f_vec[i] = points_from_map[i-n_elem];
        }        
        
        return f_vec;
    }
    
    
}

bool checkDist(const VectorXdVector& points,const int n_elem,const double dist){
    int dim = points[0].rows();
    double real_dist;
    Eigen::VectorXd distance_vec(dim);
    for(int k = 0; k < n_elem; k++){
        for(int i = n_elem-1; i >= 0; i--){
            int res = i - k;
            for(int j = 0; j < dim; j++){
                if( res == 0){
                }
                else{
                    distance_vec(j) = points[k](j)-points[i](j);
                }
            }
            real_dist = distance_vec.norm();
            if(real_dist >= dist){
            }
            else{
                return false;
            }
        }
    }
    return true;
}




Eigen::VectorXd ICP(const transf & InitialT,const VectorXdVector& obj,
        BaseTreeNode* map_tree,const double max_distance,const int iterations,
        const double kernel_threshold){
    
    
    transf T = InitialT;
    Eigen::VectorXd x(6);
    Eigen::VectorXd dx(6);
    Eigen::VectorXd alpha(3);
    alpha =  t2v(InitialT);
    x << InitialT.t(0), InitialT.t(1), InitialT.t(2), alpha(0),alpha(1),alpha(2);
    cout << x <<endl;  
    int obj_size = obj.size();
    for(int n = 0; n < iterations; n++){
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6,6); //6x6
        Eigen::VectorXd b = Eigen::RowVectorXd::Zero(6);  //6x1
        double chi = 0;
        for (int j = 0; j < obj_size; j++){
            transf EJ = errorAndJacobian(T,obj[j],map_tree,max_distance);
            chi = EJ.t.transpose()*EJ.t;
            if(chi > kernel_threshold){
                EJ.t *= sqrt(kernel_threshold/chi);
                chi = kernel_threshold;
            }
            if(n%20 == 0 && j == 0){
                cout << "mean square error: " << chi << endl; 
                cout << "iteration number "<< n <<'/'<<iterations<<endl;
                cout << "actual error " << EJ.t.transpose()<< endl;
            }
            H += EJ.R.transpose()*EJ.R;   // 6x6
            b += EJ.R.transpose()*EJ.t;    // 6x1  
        }
        //cout << "H:"<< endl << H << endl;
        //cout << "b" << endl << b.transpose() << endl;
        dx = -H.inverse()*b; // 6x6 * 6x1
        //cout << "dx:" << endl << dx.transpose() << endl;
        x += dx;    
        T = v2t(x);
        //cout << "actual transformation:"<<endl<<T.R<<endl;
    }
    return x;
}

transf v2t(const Eigen::VectorXd & a){
    
    Eigen::Matrix3d Rx, Ry, Rz;
    Eigen::VectorXd t(3);
    t << a(0), a(1), a(2);
    
    
    // define the rotation matrix w.r.t the 3 axis
    Rx << 1.0, 0.0, 0.0,0.0, cos(a(3)),sin(a(3)), 0.0,
            -sin(a(3)), cos(a(3)); 
    
    Ry << cos(a(4)), 0.0, -sin(a(4)), 0.0, 1.0, 0.0, sin(a(4)),
            0.0, cos(a(4));
    
    Rz << cos(a(5)), sin(a(5)), 0.0, -sin(a(5)), cos(a(5)), 0.0,
            0.0, 0.0, 1.0;
    
     
    transf newT;
    newT.R = Rx*Ry*Rz;
    newT.t = t;
    return newT;
}

Eigen::Vector3d t2v(const transf & t){
    Eigen::MatrixXd R = t.R;
    //assert(check_Rotation(R));
    float cy = sqrt(R(0,0)*R(0,0) + R(0,1)*R(0,1));
    bool singular = cy < 1e-6;
    double Alpha_x,Alpha_y,Alpha_z;
    Eigen::Vector3d angles;
    
    
    if (!singular){
        Alpha_x = atan2 (R(1,2),R(2,2));
        Alpha_y = atan2 (-R(0,2),cy);
        Alpha_z = atan2 (-R(0,1),R(0,0));
    }
    else{
        Alpha_x = atan2 (-R(2,1),R(1,1));
        Alpha_y = atan2 (-R(0,2),cy);
        Alpha_z = 0;
    }
    return angles = Eigen::Vector3d(Alpha_x,Alpha_y,Alpha_z);
}



Eigen::VectorXd getError(const Eigen::VectorXd & obj,
        const transf& T, BaseTreeNode* map_tree, const double max_distance){
    int dim = obj.rows();
    Eigen::VectorXd my_error;
    Eigen::VectorXd new_obj = Eigen::RowVectorXd(dim);
    
    new_obj = T.R*obj + T.t; 
    
    
    
    // check the number of inliers
    Eigen::VectorXd query_point=new_obj;
    Eigen::VectorXd answer; 
    double bf_distance=map_tree->findNeighbor(answer, query_point, max_distance);
    if (bf_distance>=0){
        my_error = new_obj-answer;
    }else{
        my_error = Eigen::Vector3d(max_distance/sqrt(3),max_distance/sqrt(3),
                max_distance/sqrt(3));
    }
    
    return my_error;
    }



transf errorAndJacobian(const transf & InitialT,
        const Eigen::VectorXd & obj, BaseTreeNode* map_tree,
        const double max_distance){
    
    transf EJ;
    Eigen::Matrix3d Rx, Ry, Rz, Rx_prime, Ry_prime, Rz_prime;
    Eigen::MatrixXd R = InitialT.R;
    Eigen::VectorXd a = t2v(InitialT);
    
    
    // define the rotation matrix w.r.t the 3 axis
    Rx << 1.0, 0.0, 0.0,0.0, cos(a(0)),sin(a(0)), 0.0,
            -sin(a(0)), cos(a(0)); 
    
    Ry << cos(a(1)), 0.0, -sin(a(1)), 0.0, 1.0, 0.0, sin(a(1)),
            0.0, cos(a(1));
    
    Rz << cos(a(2)), sin(a(2)), 0.0, -sin(a(2)), cos(a(2)), 0.0,
            0.0, 0.0, 1.0;
    
    Rx_prime << 0.0, 0.0, 0.0, 0.0, -sin(a(0)), cos(a(0)),
            0.0, -cos(a(0)), -sin(a(0));
    
    Ry_prime << -sin(a(1)), 0.0, -cos(a(1)), 0.0, 0.0, 0.0,
            -cos(a(1)), 0.0, sin(a(1));
    
    Rz_prime << -sin(a(2)), cos(a(2)), 0.0, -cos(a(2)),
            -sin(a(2)), 0.0, 0.0, 0.0, 0.0;
    
    transf newT;
    newT.R = Rx*Ry*Rz;
    newT.t = InitialT.t;
    Eigen::VectorXd my_error;
    
    my_error = getError(obj,newT, map_tree,max_distance);
    
    EJ.t = my_error;
    Eigen::Vector3d c1 = Rx_prime*Ry*Rz*obj;
    Eigen::Vector3d c2 = Rx*Ry_prime*Rz*obj;
    Eigen::Vector3d c3 = Rx*Ry*Rz_prime*obj;
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3,3);
    EJ.R = Eigen::MatrixXd::Zero(3,6);
    //compute the 3x6 jacobian 
    for(int i = 0; i < 3; i++){
        int k = i+3;
        for(int j = 0; j < 3 ; j++){
            EJ.R(i,j) = identity(i,j);
            if(k == 3){
                EJ.R(j,k) = c1(j);
            }else if( k == 4){
                EJ.R(j,k) = c2(j);
            }else if (k == 5){
                EJ.R(j,k) = c3(j);
            }
        }
    }
    
    return EJ;

            
        
    
}



// check if R is a valid rotation matrix 
bool check_Rotation(const Eigen::MatrixXd & R){
    Eigen::MatrixXd  Rt = R.transpose();
    int r = R.rows();
    int c = R.cols();
    Eigen::MatrixXd should_be_identity = R * Rt;
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(r,c);
    double val = (should_be_identity-identity).squaredNorm();
    return val < 1e-6;
  
}

transf read_transf_file(string out_file,int buff_size){
    int cols = 0, rows = 0;
    double buff[buff_size];
    ifstream infile;
    infile.open(out_file);
    while (! infile.eof()){
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }



    infile.close();
    rows --;
    transf FinalT;
    Eigen::MatrixXd R(rows-1,cols);
    Eigen::VectorXd t(cols);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            if (i < 3){
                R(i,j) = buff[ cols*i+j ];
            }
            else{
                t(j) = buff[cols*i +j ];
            }
        }
    }
    FinalT.R = R;
    FinalT.t = t;
    return FinalT;
}





void draw_distance_map(const VectorXdVector& map, const VectorXdVector& obj){
    double x = 0, y = 0, z = 0;
    vector<Point3d> pts_map;
    vector<Point3d> pts_object;
    viz::Viz3d window; //creating a Viz window
    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", viz::WCoordinateSystem(1.0));
    for(int i = 0; i < map.size(); i ++){
        x = map[i](0);
        y = map[i](1);
        z = map[i](2);
        pts_map.push_back(Point3d(x,y,z));
    }
    for(int i = 0; i < obj.size(); i += 3){
        x = obj[i](0);
        y = obj[i](1);
        z = obj[i](2);
        pts_object.push_back(Point3d(x,y,z));        
    }
    //Displaying the 3D points in green
    window.showWidget("pts_map", viz::WCloud(pts_map, viz::Color::red()));
    window.showWidget("pts_object", viz::WCloud(pts_object, viz::Color::blue()));    
    window.spin();
}

/*void draw_distance_map(vector<double> point_map){
    double x = 0, y = 0, z = 0;
    vector<Point3f> pts3d;
    viz::Viz3d window; //creating a Viz window
    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", viz::WCoordinateSystem(1.0));
    for(int i = 0; i < point_map.size(); i += 3){
        x = point_map[i];
        y = point_map[i + 1];
        z = point_map[i + 2];
        pts3d.push_back(Point3f(x,y,z));
    }
    window.showWidget("points", viz::WCloud(pts3d, viz::Color::red()));
    window.spin();
}*/

