#ifndef MYHEADER_H
#define MYHEADER_H
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Eigenvalues> 
#include <Eigen/SVD>
#include <math.h>
#include <time.h>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widgets.hpp>

#define PI 3.14159265

typedef std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> >
VectorXdVector;
typedef struct{
    Eigen::MatrixXd R;
    Eigen::VectorXd t;
}transf;
VectorXdVector get_points(std::string, int);
/**
   BaseTreeNode class. Represents a base class for a node in the search tree.
   It hasa dimension (given on construction).
*/
class BaseTreeNode{
    protected:
        int _dimension;
    public:
        BaseTreeNode(int dimension_);
        virtual ~BaseTreeNode();
        virtual double findNeighbor(Eigen::VectorXd& answer, 
			      const Eigen::VectorXd& query, 
			      const double max_distance) const = 0;
        inline int dimension() const {return _dimension;}
    
};
/**
   Leaf node: it contains a vector of points on which.
   The search function performs a linear search in the list
*/

class LeafNode: public BaseTreeNode{
    protected:
        VectorXdVector _points;
    public:
        LeafNode(int dimension_);
        const VectorXdVector& points() const;
        VectorXdVector& points();
        virtual double findNeighbor(Eigen::VectorXd& answer,
			      const Eigen::VectorXd& query, 
			      const double max_distance) const;
        
};
/**
   Middle node: it represents a splitting plane, and has 2 child nodes,
   that refer to the set of points to the two sides of the splitting plane.
   A splitting plane is parameterized as a point on a plane and as a normal to the plane.
*/
class MiddleNode: public BaseTreeNode{
    protected:
        Eigen::VectorXd _normal;
        Eigen::VectorXd _mean;
        BaseTreeNode* _left_child;
        BaseTreeNode* _right_child;
    public:
          MiddleNode(int dimension_,
	     const Eigen::VectorXd& mean_,
	     const Eigen::VectorXd& normal_,
	     BaseTreeNode* left_child,
	     BaseTreeNode* right_child);
          virtual ~MiddleNode();
          inline const Eigen::VectorXd& mean() const {return _mean;};
          inline const Eigen::VectorXd& normal() const {return _normal;};
          bool side(const Eigen::VectorXd& query_point) const;
          virtual double findNeighbor(Eigen::VectorXd& answer,
		      const Eigen::VectorXd& query, 
		      const double max_distance) const;
};

double splitPoints(Eigen::VectorXd& , Eigen::VectorXd&,
		   VectorXdVector&, VectorXdVector& , 
		   const VectorXdVector& );
BaseTreeNode* buildTree(const VectorXdVector& , double);

// take as input 2 vector and return N random points from both vector. These 3
// points have to be distant at least "dist" from each other
VectorXdVector enhancedRandomPick(const VectorXdVector& ,const VectorXdVector& , 
            int , double );

// Take N random points from both the point clouds, return a vector of 
// 2N length, where the first N elements are from point cloud 1 and the others
// are from the second one
VectorXdVector randomPick(const VectorXdVector&, const VectorXdVector&, int);


// ICP with linear relaxation: 
// inputs: 2 vector of points 
// outputs: transformation matrix between the 2 vector.

transf ICPLinearRelaxation(const VectorXdVector&,
        const VectorXdVector& );


// ICP without linear relaxation:

Eigen::VectorXd ICP(const transf &,const VectorXdVector& ,BaseTreeNode*,
        const double,const int,const double);

transf errorAndJacobian(const transf &, const Eigen::VectorXd&,
        BaseTreeNode* ,const double);

Eigen::VectorXd getError(const Eigen::VectorXd&, const transf&,
        BaseTreeNode*, const double);

Eigen::Vector3d t2v(const transf &);

bool check_Rotation(const Eigen::MatrixXd &);

int getScore(const VectorXdVector&, const VectorXdVector&, const transf&,
        BaseTreeNode*, const double);


// check if distance from points greater or equal than "dist"
bool checkDist(const VectorXdVector&,const int n_elem,const double);


transf v2t(const Eigen::VectorXd &);

void draw_distance_map(const VectorXdVector& , const VectorXdVector& );


transf read_transf_file(std::string,int);
#endif /* MYHEADER_H */

