// Compile command: g++ -Wall -Wextra -std=c++17 -O2 -pthread -I/usr/include/python3.8 -o triangular_cone triangular_cone.cpp -lpython3.8
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <utility>
#include <functional>
#include <thread>
#include <map>
#include <string>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include "matplotlibcpp.h"


namespace plt = matplotlibcpp;
using Eigen::MatrixXd;
using Eigen::Matrix2cd;
using Eigen::Vector2cd;
using std::vector;
using std::sqrt;
using std::cos;
using std::acos;
using std::sin;
using std::atan2;
using namespace std::complex_literals;

typedef std::complex<double> cd;
typedef vector<vector<vector<cd>>> grid_t;
typedef std::function<Matrix2cd(int, double, double)> gen_coin_t;

// Settings here
int num_steps = 1000;
int initialState = 0; // See main function for details
vector<std::string> initialStateName = {"square", "shiftright", "shiftdl", "center", "bothsides"};
double eps = 0.01;
double dy = sqrt(3);
bool bugLambda = true; // Set to true to set Lambda = I_2
bool plotCone = true; // Set to true to plot the results in the cone coordinates
bool showVectField = true; // Set to true to show the vector field in original coordinates
int ymin = -50, ymax = 50;
int xmin = 2*ymin-2, xmax = 2*ymax+2;
int ntriangles_x = xmax-xmin+1;
int ntriangles_y = ymax-ymin+1;
int center[2] = {-xmin, -ymin};
std::string prefix;

std::time_t now = time(0);
std::tm *ltm = localtime(&now);

grid_t zerogrid() {
    return vector<vector<vector<cd>>>(ntriangles_x, vector<vector<cd>> (ntriangles_y, vector<cd> (3, 0. + 0i)));
}
grid_t nangrid() {
    return vector<vector<vector<cd>>>(ntriangles_x, vector<vector<cd>> (ntriangles_y, vector<cd> (3, std::nan("") + 0i)));
}

grid_t grid = zerogrid();

Matrix2cd H, Q;

double sumAmplitudes() {
    double ret = 0.0;
    for(int x = xmin; x <= xmax; x++)
        for(int y = ymin; y <= ymax; y++)
            for(int side = 0; side < 3; side++) {
                cd val = grid[x+center[0]][y+center[1]][side];
                ret += std::real(val*std::conj(val));
            }
    return ret;
}

// Simplifies initialization for some initial states
void normalizeGrid() {
    double target = sumAmplitudes();
    double mul = 1/sqrt(target);
    if(target == 0)
        return;
    for(int x = xmin; x <= xmax; x++)
        for(int y = ymin; y <= ymax; y++)
            for(int side = 0; side < 3; side++)
                grid[x+center[0]][y+center[1]][side] *= mul;
}

inline int modulo(int a, int b) {
    return (a%b+b)%b;
}

void init_HQ() {
    H << 1, 1, 1, -1;
    H /= sqrt(2);
    Q << 1, -1i, 1, 1i;
    Q /= sqrt(2);
}

Matrix2cd gen_Ui(double thetai) {
    Matrix2cd ret;
    ret << 
        cos(thetai/2), sin(thetai/2),
        -sin(thetai/2), cos(thetai/2);
    return ret;
}

inline int sign(double d) {
    if(d>=0)
        return 1;
    else
        return -1;
}

inline double sq(double d) {
    return d*d;
}

inline double correct_fmod(const double a, const double b) {
    return std::fmod(std::fmod(a,b)+b, b);
}

inline double principal_measure(const double theta) {
    double ret = correct_fmod(theta, 2*M_PI);
    if(ret > M_PI)
        ret -= 2*M_PI;
    return ret;
}

Matrix2cd lamb(double rx, double ry) {
    Matrix2cd ret;
    if(bugLambda)
        ret = Matrix2cd::Identity();
    else if(rx == 0 && ry == 0) {
        ret <<
            6./5.*cos(-5.*M_PI/6.), -6./5.*sin(-5.*M_PI/6.),
            6./5.*sin(-5.*M_PI/6.), 5./5.*cos(-5.*M_PI/6.);
    }
    else {
        double r = 5./6.*sqrt(rx*rx+ry*ry);
        double thetap = atan2(ry, rx);
        double theta2 = principal_measure(thetap + 5*M_PI/6);
        double theta = 6./5.*theta2;
        double x = r*cos(theta), y = r*sin(theta);
        ret <<
            6./5.*(x/r*cos(thetap) + y/r*sin(thetap)),
            6./5.*(y/r*cos(thetap) - x/r*sin(thetap)),
            6./5.*(x/r*sin(thetap) - y/r*cos(thetap)),
            6./5.*(y/r*sin(thetap) + x/r*cos(thetap));
    }
    return ret;
}

vector<cd> l(double rx, double ry) {
    Matrix2cd lam = lamb(rx, ry);
    double sqrt3 = sqrt(3.0);
    return {lam(0,0), lam(1,0)/sqrt3, -lam(1,0)/sqrt3, lam(0,1), lam(1,1)/sqrt3, -lam(1,1)/sqrt3};
}

double gen_theta(int i, double rx, double ry) {
    return std::real(M_PI/2 + sqrt(eps)*l(rx, ry)[i]);
}

Matrix2cd gen_U(int i, double rx, double ry) {
    return gen_Ui(gen_theta(i, rx, ry));
}

Matrix2cd gen_Ubis(int i, double rx, double ry) {
    return gen_U(i+3, rx, ry);
}

Matrix2cd gen_Ustar(int i, double rx, double ry) {
    return gen_U(i, rx, ry).adjoint();
}

Matrix2cd gen_Ubisstar(int i, double rx, double ry) {
    return gen_Ustar(i+3, rx, ry);
}

grid_t shift(grid_t grid) {
    grid_t ngrid = zerogrid();
    for(int x = xmin; x <= xmax; x++) {
        for(int y = ymin; y <= ymax; y++) {
            for(int i = 0; i < 3; i++) {
                int iprec = ((i-1)%3+3)%3;
                ngrid[x+center[0]][y+center[1]][i] = grid[x+center[0]][y+center[1]][iprec];
            }
        }
    }
    return ngrid;
}

std::pair<double, double> real_coords(int iside, int x, int y, bool show = false) {
    double dec;
    if(show)
        dec = .4;
    else
        dec = .5;
    double xcoord, ycoord;
    if((x+y)%2==0) {
        if(iside == 0) {
            xcoord = x-dec;
            ycoord = (y+.5)*dy;
        }
        else if(iside == 1) {
            xcoord = x+dec;
            ycoord = (y+.5)*dy;
        }
        else {
            xcoord = x;
            ycoord = (y+.5+dec)*dy;
        }
    }
    else {
        if(iside == 0) {
            xcoord = x+dec;
            ycoord = (y+.5)*dy;
        }
        else if(iside == 1) {
            xcoord = x-dec;
            ycoord = (y+.5)*dy;
        }
        else {
            xcoord = x;
            ycoord = (y+.5-dec)*dy;
        }
    }
    return std::make_pair(sqrt(eps)*xcoord, sqrt(eps)*ycoord);
}

std::pair<double, double> cone_coords(int iside, int x, int y, bool show = false) {
    double rx, ry;
    std::tie(rx, ry) = real_coords(iside, x, y, show);
    double r = 5./6.*sqrt(rx*rx+ry*ry);
    double theta_before = atan2(ry, rx);
    double theta_rotate = principal_measure(theta_before+5*M_PI/6.);
    double theta = 6./5.*theta_rotate;
    //std::cerr << "theta " << theta_before*180/M_PI << " -> " << theta_rotate*180/M_PI << " -> " << theta*180/M_PI << std::endl;
    return std::make_pair(r*cos(theta), r*sin(theta));
}

const int DELTAS[][2] = {{-1,0}, {1,0}, {0,1}};
const int NUM_THREADS = 8;

void applyCoinsPartial(grid_t &ngrid, grid_t &grid, gen_coin_t &gen_coin, int loc_xmin, int loc_xmax) {
    for(int x = loc_xmin; x < loc_xmax; x++) {
        for(int y = ymin; y <= ymax; y++) {
            if((x+y)%2 || (x>y && y>=0))
                continue;
            for(int iside = 0; iside < 3; iside++) {
                cd thisval = grid[x+center[0]][y+center[1]][iside];
                int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
                if(xo > xmax || xo < xmin || yo > ymax || yo < ymin || (xo > yo && yo >= 0)) { // pas de propagation aux bords
                    ngrid[x+center[0]][y+center[1]][iside] = thisval;
                    continue;
                }
                cd otherval = grid[modulo(xo+center[0], ntriangles_x)][modulo(yo+center[1], ntriangles_y)][iside];
                Vector2cd vect;
                vect << thisval, otherval;
                double rx, ry;
                std::tie(rx, ry) = real_coords(iside, x, y);
                Matrix2cd coin = gen_coin(iside, rx, ry);
                Vector2cd newvect = coin*vect;
                ngrid[x+center[0]][y+center[1]][iside] = newvect(0);
                ngrid[modulo(xo+center[0], ntriangles_x)][modulo(yo+center[1], ntriangles_y)][iside] = newvect(1);
            }
        }
    }
}

grid_t applyCoins(grid_t grid, gen_coin_t gen_coin, bool multithread = true) {
    grid_t ngrid = nangrid();
    if(!multithread) {
        for(int x = xmin; x <= xmax; x++) {
            for(int y = ymin; y <= ymax; y++) {
                if((x+y)%2 || (x>y && y>=0)) // pentagone : on enlève un bout de 60°
                    continue;
                for(int iside = 0; iside < 3; iside++) {
                    cd thisval = grid[x+center[0]][y+center[1]][iside];
                    int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
                    if(xo > xmax || xo < xmin || yo > ymax || yo < ymin || (xo > yo && yo >= 0)) { // pas de propagation aux bords
                        ngrid[x+center[0]][y+center[1]][iside] = thisval;
                        continue;
                    }
                    cd otherval = grid[modulo(xo+center[0], ntriangles_x)][modulo(yo+center[1], ntriangles_y)][iside];
                    Vector2cd vect;
                    vect << thisval, otherval;
                    double rx, ry;
                    std::tie(rx, ry) = real_coords(iside, x, y);
                    Matrix2cd coin = gen_coin(iside, rx, ry);
                    Vector2cd newvect = coin*vect;
                    ngrid[x+center[0]][y+center[1]][iside] = newvect(0);
                    ngrid[modulo(xo+center[0], ntriangles_x)][modulo(yo+center[1], ntriangles_y)][iside] = newvect(1);
                }
            }
        }
    }
    else {
        std::thread threads[NUM_THREADS];
        int delta_x = ntriangles_x/NUM_THREADS;
        for(int iThread = 0; iThread < NUM_THREADS-1; iThread++) {
            threads[iThread] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(gen_coin), xmin+iThread*delta_x, xmin+(iThread+1)*delta_x);
        }
        threads[NUM_THREADS-1] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(gen_coin), xmin+(NUM_THREADS-1)*delta_x, xmax+1);
        for(int iThread = 0; iThread < NUM_THREADS; iThread++)
            threads[iThread].join();
    }
    // We may miss some sides of 1 (tip up) triangles which are not adjacent to a valid 0 triangle
    for(int x = xmin; x <= xmax; x++)
        for(int y = ymin; y <= ymax; y++)
            for(int side = 0; side < 3; side++) {
                cd &newval = ngrid[x+center[0]][y+center[1]][side];
                if(std::isnan(std::real(newval)))
                    newval = grid[x+center[0]][y+center[1]][side];
            }
    return ngrid;
}

void plotVectorField(double minx, double maxx, double miny, double maxy, int gridstep = 20) {
    vector<double> xloc, yloc;
    vector<double> vectx, vecty;
    double dx = (maxx-minx)/gridstep, dy = (maxy-miny)/gridstep;
    for(double x = minx; x <= maxx; x += dx)
        for(double y = miny; y <= maxy; y += dy) {
            for(int i = 0; i < 2; i++) {
                xloc.push_back(x);
                yloc.push_back(y);
                Matrix2cd deform = lamb(x,y);
                vectx.push_back(std::real(deform(0,i)));
                vecty.push_back(std::real(deform(1,i)));
            }
        }
    plt::quiver(xloc, yloc, vectx, vecty, {{"pivot","tail"}, {"color", "grey"}});
}

void plot(int iGrid = -1) {
    PyObject *fig;
    if(plotCone) {
        fig = plt::figure_size(1000,1000);
        plt::xlim(xmin*sqrt(eps), xmax*sqrt(eps));
        plt::ylim(ymin*sqrt(eps)*dy, ymax*sqrt(eps)*dy);
        plt::set_aspect_equal();
        vector<double> xlist, ylist, colorlist;
        vector<double> listvals;
        for(int x = xmin; x <= xmax; x++) {
            for(int y = ymin; y <= ymax; y++) {
                for(int iside = 0; iside < 3; iside++) {
                    cd val = grid[x+center[0]][y+center[1]][iside];
                    double col = std::real(val*std::conj(val));
                    listvals.push_back(col);
                }
            }
        }
        std::sort(listvals.rbegin(), listvals.rend());
        double maxi = (listvals[0]+listvals[1])/2;
        double threshold = maxi/10;
        for(int x = xmin; x <= xmax; x++) {
            for(int y = ymin; y <= ymax; y++) {
                for(int iside = 0; iside < 3; iside++) {
                    cd val = grid[x+center[0]][y+center[1]][iside];
                    double col = std::real(val*std::conj(val));
                    if(col > threshold) {
                        double rx, ry;
                        std::tie(rx, ry) = cone_coords(iside, x, y, true);
                        xlist.push_back(rx);
                        ylist.push_back(ry);
                        colorlist.push_back(col);
                    }
                }
            }
        }
        if(maxi == 0.0)
            maxi = 1.0;
        maxi *= 0.6;
        plt::scatter_colored(xlist, ylist, colorlist, 5, {{"cmap","hot_r"}, {"vmin", "0"}, {"vmax", std::to_string(maxi)}});
    }
    else {
        fig = plt::figure_size(1000,1000);
        plt::set_aspect_equal();
        vector<vector<double>> imgrid(ymax-ymin+1, vector<double>(xmax-xmin+1, 0.0));
        vector<double> listvals;
        for(int y = ymin; y <= ymax; y++) {
            for(int x = xmin; x <= xmax; x++) {
                double sum = 0.0;
                for(int iside = 0; iside < 3; iside++) {
                    cd val = grid[x+center[0]][y+center[1]][iside];
                    double col = std::real(val*std::conj(val));
                    sum += col;
                }
                listvals.push_back(sum);
                imgrid[y+center[1]][x+center[0]] = sum;
            }
        }
        std::sort(listvals.rbegin(), listvals.rend());
        double maxi = (listvals[0]+listvals[1])/2;
        if(maxi == 0.0)
            maxi = 1.0;
        maxi *= .6;
            
        double minx = sqrt(eps)*(xmin-.5);
        double maxx = sqrt(eps)*(xmax+.5);
        double miny = sqrt(eps)*dy*(ymin-.5);
        double maxy = sqrt(eps)*dy*(ymax+.5);
        plt::imshow(imgrid, {minx, maxx, miny, maxy}, {{"origin", "lower"}, {"cmap", "hot_r"}, {"vmin", "0.0"}, {"vmax", std::to_string(maxi)}});
        plt::plot({0.0, maxy/tan(M_PI/3)}, {0.0, maxy}, {{"color","red"}});
        plt::plot({0.0, xmax*sqrt(eps)}, {0.0, 0.0}, {{"color","red"}});
        if(showVectField)
            plotVectorField(xmin*sqrt(eps), xmax*sqrt(eps), ymin*sqrt(eps)*dy, ymax*sqrt(eps)*dy, 20);
    }
    std::ostringstream filename;
    filename << prefix << "_" << iGrid << ".png";
    plt::save(prefix + "/" + filename.str());
    plt::clf();
    plt::close(fig);
    Py_DECREF(fig);

    // Plot around dislocation line
    vector<double> xlist2, ylist2;
    for(int x = 0; x <= xmax; x++) {
        vector<int> klist;
        if((x+1)%2)
            klist = {1, 2, 0};
        else
            klist = {0, 2, 1};
        for(int k : klist) {
            double rx, ry;
            std::tie(rx, ry) = real_coords(k, x, -1);
            xlist2.push_back(rx);
            cd val = grid[x+center[0]][center[1]-1][k];
            double col = std::real(val*std::conj(val));
            ylist2.push_back(col);
        }
    }
    plt::plot(xlist2, ylist2);
    std::ostringstream filename_disloc;
    filename_disloc << prefix << "_dislocation_" << iGrid << ".png";
    plt::save(prefix + "/" + filename_disloc.str());
    plt::clf();
    plt::close();
}

void step_walk(int step = -1) {
    std::cerr << "Begin step " << step << std::endl;
    grid = applyCoins(grid, [](int i, double rx, double ry) {
        Matrix2cd ret = H;
        return ret;
    });
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_Ustar);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_U(((i-1)%3+3)%3, rx, ry);
            return ret;
        });
    }
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_U);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_Ustar(((i-1)%3+3)%3, rx, ry);
            return ret;
        });
    }
    grid = applyCoins(grid, [](int i, double rx, double ry) {
        Matrix2cd ret = Q*H;
        return ret;
    });
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_Ubisstar);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_Ubis(((i-1)%3+3)%3, rx, ry);
            return ret;
        });
    }
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_Ubis);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_Ubisstar(((i-1)%3+3)%3, rx, ry);
            return ret;
        });
    }
    grid = applyCoins(grid, [](int i, double rx, double ry) {
        Matrix2cd ret = Q.adjoint();
        return ret;
    });
    std::cerr << "Total amplitude: " << sumAmplitudes() << "\n";
    //normalizeGrid();
    //std::cerr << "After normalization: " << sumAmplitudes() << "\n";
    std::cerr << "End step " << step << std::endl;
}

void print_params() {
    std::ofstream ostream(prefix + "/settings.txt");
    ostream << "num_steps = " << num_steps << "\n";
    ostream << "eps = " << eps << "\n";
    ostream << "xmin = " << xmin << "\n";
    ostream << "xmax = " << xmax << "\n";
    ostream << "ymin = " << ymin << "\n";
    ostream << "ymax = " << ymax << "\n";
    if(bugLambda)
        ostream << "Lambda = I_2\n";
    else
        ostream << "Lambda != I_2\n";
    if(plotCone)
        ostream << "Cone coordinates\n";
    else
        ostream << "Grid coordinates\n";
    ostream.close();
    //std::cout << "Deformation matrix at (-1,1):\n" << lamb(-1,1) << "\n";
    /*for(int i = 0; i < 6; i++) {
        std::cout << "U" << i << "(1,1):\n" << gen_U(i,1,1) << "\n";
    }*/
    //std::cout << "H=\n" << H << "\n";
    //std::cout << "Q=\n" << Q << std::endl;
}

void listInitialStates() {
    for(size_t i = 0; i < initialStateName.size(); i++)
        std::cerr << "- " << i << ": " << initialStateName[i] << "\n";
}

int main(int argc, char **argv)
{
    if(argc <= 2) {
        std::cerr << "Usage: " << std::string(argv[0]) << " <initial state> <plot cone>\n";
        std::cerr << "Initial states:\n";
        listInitialStates();
        std::cerr << "<plot cone> should be 1 if the figure should be plotted in cone coordinates, 0 otherwise.\n";
        return 1;
    }
    initialState = std::atoi(argv[1]);
    if(initialState < 0 || initialState >= (int)initialStateName.size()) {
        std::cerr << "Invalid initial state " << initialState << ". List of possible states:\n";
        listInitialStates();
        return 2;
    }
    plotCone = (bool) std::atoi(argv[2]);
    std::ostringstream str;
    str <<
        "simul_cone_" << initialStateName[initialState];
    if(plotCone)
        str << "_conecoord_";
    else
        str << "_altcoord_";
    str <<
        std::setw(4) << std::setfill('0') << ltm->tm_year+1900 << "-" << 
        std::setw(2) << ltm->tm_mon+1 << "-" << 
        std::setw(2) << ltm->tm_mday << "_" << 
        std::setw(2) << ltm->tm_hour << "-" << 
        std::setw(2) << ltm->tm_min << "-" << 
        std::setw(2) << ltm->tm_sec;
    prefix = str.str();
    std::filesystem::create_directory(prefix);
    init_HQ();
    print_params();
    // Initial state
    if(initialState == 0) {
        for(int x = xmin/5; x <= xmax/5; x++)
            for(int y = ymin/5; y <= ymax/5; y++)
                for(int k = 0; k < 3; k++)
                    if(x <= y || y < 0)
                        grid[center[0]+x][center[1]+y][k] = 1;
        normalizeGrid();
    }
    // Shifted
    else if(initialState == 1) {
        for(int k = 0; k < 3; k++)
            grid[center[0]+xmax/2-1][center[1]-1][k]=1/sqrt(3);
    }
    // Shifted (in another way)
    else if(initialState == 2) {
        for(int k = 0; k < 3; k++)
            grid[center[0]+xmin/4+1][center[1]+ymin/4+1][k]=1/sqrt(3);
    }
    // Centered
    else if(initialState == 3) {
        vector<vector<int>> centercoord = {{-1,0},{0,0},{-1,-1},{0,-1},{1,-1}};
        for(vector<int> &coord : centercoord)
            for(int k = 0; k < 3; k++)
                grid[center[0]+coord[0]][center[1]+coord[1]][k]=1/sqrt(3*centercoord.size());
    }
    // Shifted, both sides of the line
    else if(initialState == 4) {
        for(int k=0; k<3; k++) {
            grid[center[0]+10][center[1]+10][k] = 1/sqrt(6);
            grid[center[0]+21][center[1]-1][k] = 1/sqrt(6);
        }
    }
    plot(0);
    for(int i = 0; i < num_steps; i++) {
        step_walk(i);
        if((i+1)%10 == 0)
            plot(i+1);
    }
}
