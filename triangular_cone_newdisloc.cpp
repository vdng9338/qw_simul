// Compile command: g++ -Wall -Wextra -std=c++17 -O2 -pthread -I/usr/include/python3.8 -I/usr/include/eigen3 -o triangular_cone_newdisloc triangular_cone_newdisloc.cpp -lpython3.8
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
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
double dy = sqrt(3);
bool plotCone = true; // Set to true to plot the results in the cone coordinates
int ymin = -200, ymax = 200;
int xmin = 3*ymin/2-ymin%2-1, xmax = 3*ymax/2+ymax%2+1;
int ntriangles_x = xmax-xmin+1;
int ntriangles_y = ymax-ymin+1;
int center[2] = {-xmin, -ymin};
std::string prefix;
int initialState = 0;
vector<std::string> initialStateName = {"square", "shifttop", "shiftdl", "center", "bothsides", "everywhere"};

std::time_t now = time(0);
std::tm *ltm = localtime(&now);

grid_t zerogrid() {
    return vector<vector<vector<cd>>>(ntriangles_x, vector<vector<cd>> (ntriangles_y, vector<cd> (3, 0. + 0i)));
}
grid_t nangrid() {
    return vector<vector<vector<cd>>>(ntriangles_x, vector<vector<cd>> (ntriangles_y, vector<cd> (3, std::nan("") + 0i)));
}

grid_t grid = zerogrid();

Matrix2cd U, C0;
Matrix2cd sigmay, sigmaz;

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
    sigmaz <<
        1,0,
        0,-1;
    sigmay <<
        0, -1.0i,
        1.0i, 0;
    C0 = std::exp(1.0i*M_PI/3.0) * ((1.0i*M_PI/3.0)*sigmaz).exp();
    double alpha = -std::acos(sqrt(5)/3);
    Matrix2cd U0 = (-1.0i*alpha*sigmay/2.0).exp();
    U = U0*C0*C0;
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

inline double sqmodulus(const std::complex<double> comp) {
    return std::real(comp*std::conj(comp));
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
    // Dislocation: particular case
    if(x == 0 && y >= 0) { // Vertical line
        if((x+y)%2 == 0) { // Tip down
            if(iside == 0) {
                xcoord = x-dec;
                ycoord = (y+.5)*dy;
            }
            else if(iside == 1) {
                xcoord = x;
                ycoord = (y+.5)*dy;
            }
            else {
                xcoord = x-dec;
                ycoord = (y+.5+dec)*dy;
            }
        }
        else { // Tip up
            if(iside == 0) {
                xcoord = x;
                ycoord = (y+.5)*dy;
            }
            else if(iside == 1) {
                xcoord = x-dec;
                ycoord = (y+.5)*dy;
            }
            else {
                xcoord = x-dec;
                ycoord = (y+.5-dec)*dy;
            }
        }
    }
    else if(y >= 0 && (x == 3*y+1 || x == 3*y+2)) { // Diagonal line
        if((x+y)%2 == 0) { // Tip down
            if(iside == 0) {
                xcoord = x-dec/2.;
                ycoord = (y+.5-dec/2.)*dy;
            }
            else if(iside == 1) {
                xcoord = x+dec;
                ycoord = (y+.5)*dy;
            }
            else {
                xcoord = x+dec/2.;
                ycoord = (y+.5+dec/2.)*dy;
            }
        }
        else { // Tip up
            if(iside == 0) {
                xcoord = x+dec+dec/2.;
                ycoord = (y+.5-dec/2.)*dy;
            }
            else if(iside == 1) {
                xcoord = x-dec/2.;
                ycoord = (y+.5-dec/2.)*dy;
            }
            else {
                xcoord = x;
                ycoord = (y+.5-dec)*dy;
            }
        }
    }
    else if((x+y)%2==0) {
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
    return std::make_pair(xcoord, ycoord);
}

std::pair<double, double> cone_coords(int iside, int x, int y, bool show = false) {
    double rx, ry;
    std::tie(rx, ry) = real_coords(iside, x, y, show);
    double r = 5./6.*sqrt(rx*rx+ry*ry);
    double theta_before = atan2(ry, rx);
    double theta_rotate = principal_measure(theta_before+4*M_PI/6.);
    double theta = 6./5.*theta_rotate;
    return std::make_pair(r*cos(theta), r*sin(theta));
}

const int DELTAS[][2] = {{1,0}, {-1,0}, {0,-1}};
const int NUM_THREADS = 8;

/**
 * For the new dislocation: 
 * - triangles (0,y), y >= 0, have triangle (3*y//2 + y%2 + 1, y//2) as a neighbor. 
 *   Since we only consider tip up triangles, with (x+y)%2 == 1, we simply need to know that
 *   triangles (0, y), y >= 0 and y odd, have side 2 of triangle (3*(y-1)/2+2, (y-1)/2) as a neighbor at side 0.
 * - triangles (3y+1, y), y >= 0 have side 1 of triangle (0, 2y) as a neighbor at side 1.
 */
void applyCoinsPartial(grid_t &ngrid, grid_t &grid, const Matrix2cd &coin, int loc_xmin, int loc_xmax) {
    for(int x = loc_xmin; x < loc_xmax; x++) {
        for(int y = ymin; y <= ymax; y++) {
            if((x+y)%2==0 || (x>0 && y>(x-1)/3))
                continue;
            for(int iside = 0; iside < 3; iside++) {
                int otherside = iside;
                cd thisval = grid[x+center[0]][y+center[1]][iside];
                int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
                if(xo > xmax || xo < xmin || yo > ymax || yo < ymin) { // No propagation at borders
                    ngrid[x+center[0]][y+center[1]][iside] = thisval;
                    continue;
                }
                if(x == 0 && y >= 0 && iside == 0) {
                    xo = 3*(y-1)/2+2;
                    yo = (y-1)/2;
                    otherside = 2;
                }
                else if(y >= 0 && x == 3*y+1 && iside == 1) {
                    xo = 0;
                    yo = 2*y;
                    otherside = 1;
                }
                cd otherval = grid[xo+center[0]][yo+center[1]][otherside];
                Vector2cd vect;
                vect << thisval, otherval;
                Vector2cd newvect = coin*vect;
                ngrid[x+center[0]][y+center[1]][iside] = newvect(0);
                ngrid[xo+center[0]][yo+center[1]][otherside] = newvect(1);
            }
        }
    }
}

grid_t applyCoins(grid_t grid, const Matrix2cd &coin, bool multithread = true) {
    grid_t ngrid = nangrid();
    if(!multithread) {
        for(int x = xmin; x <= xmax; x++) {
            for(int y = ymin; y <= ymax; y++) {
                if((x+y)%2==0 || (x>0 && y>(x-1)/3)) // dislocation : remove a 60-degree part of the lattie
                    continue;
                for(int iside = 0; iside < 3; iside++) {
                    int otherside = iside;
                    cd thisval = grid[x+center[0]][y+center[1]][iside];
                    int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
                    if(xo > xmax || xo < xmin || yo > ymax || yo < ymin) { // No propagation at borders
                        ngrid[x+center[0]][y+center[1]][iside] = thisval;
                        continue;
                    }
                    if(x == 0 && y >= 0 && iside == 0) {
                        xo = 3*(y-1)/2+2;
                        yo = (y-1)/2;
                        otherside = 2;
                    }
                    else if(y >= 0 && x == 3*y+1 && iside == 1) {
                        xo = 0;
                        yo = 2*y;
                        otherside = 1;
                    }
                    cd otherval = grid[xo+center[0]][yo+center[1]][otherside];
                    Vector2cd vect;
                    vect << thisval, otherval;
                    Vector2cd newvect = coin*vect;
                    ngrid[x+center[0]][y+center[1]][iside] = newvect(0);
                    ngrid[xo+center[0]][yo+center[1]][otherside] = newvect(1);
                }
            }
        }
    }
    else {
        std::thread threads[NUM_THREADS];
        int delta_x = ntriangles_x/NUM_THREADS;
        for(int iThread = 0; iThread < NUM_THREADS-1; iThread++) {
            threads[iThread] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(coin), xmin+iThread*delta_x, xmin+(iThread+1)*delta_x);
        }
        threads[NUM_THREADS-1] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(coin), xmin+(NUM_THREADS-1)*delta_x, xmax+1);
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

void plot(int iGrid = -1) {
    PyObject *fig;
    // Code to write the plot as Python matplotlib instructions. Use this if there are too many visible points to plot, since matplotlib-cpp segfaults in that case
    /*std::ostringstream scriptpath;
    scriptpath << prefix << "/script_" << iGrid << ".py";
    std::ofstream pyscript(scriptpath.str());
    pyscript << "from matplotlib import pyplot as plt\n";
    pyscript << "plt.figure(figsize=(10,10))\n";
    pyscript << "plt.xlim(" << xmin << "," << xmax << ")\n";
    pyscript << "plt.ylim(" << ymin*dy << "," << ymax*dy << ")\n";*/
    if(plotCone) {
        fig = plt::figure_size(1000,1000);
        plt::xlim(xmin, xmax);
        plt::ylim(ymin*dy, ymax*dy);
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
        // Continued
        /*
        pyscript << "plt.scatter([";
        for(double d : xlist)
            pyscript << d << ",";
        pyscript << "],[";
        for(double d : ylist)
            pyscript << d << ",";
        pyscript << "], c=[";
        for(double c : colorlist)
            pyscript << c << ",";
        pyscript << "], cmap=\"gist_heat_r\", vmin=0, vmax=" << maxi << ")\n";*/
        plt::scatter_colored(xlist, ylist, colorlist, 1, {{"cmap","gist_heat_r"}, {"vmin", "0"}, {"vmax", std::to_string(maxi)}});
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
            
        double minx = (xmin-.5);
        double maxx = (xmax+.5);
        double miny = dy*(ymin-.5);
        double maxy = dy*(ymax+.5);
        plt::imshow(imgrid, {minx, maxx, miny, maxy}, {{"origin", "lower"}, {"cmap", "gist_heat_r"}, {"vmin", "0.0"}, {"vmax", std::to_string(maxi)}});
        plt::plot({0.0, 0.0}, {0.0, maxy}, {{"color","red"}});
        plt::plot({0.0, maxx}, {0.0, maxx*std::tan(M_PI/6)}, {{"color","red"}});
    }
    // Continued
    /*pyscript << "plt.savefig(\"" << prefix << "_" << iGrid << ".png\")\n";
    pyscript.flush();
    pyscript.close();*/
    std::ostringstream filename;
    filename << prefix << "_" << iGrid << ".png";
    plt::save(prefix + "/" + filename.str());
    plt::clf();
    plt::close(fig);
    Py_DECREF(fig);

    // Plot around dislocation line
    vector<double> xlist2, ylist2;
    for(int y = 0; y <= ymax; y++) {
        vector<int> klist;
        if(y%2)
            klist = {2, 0, 1};
        else
            klist = {0, 1, 2};
        for(int k : klist) {
            double rx, ry;
            std::tie(rx, ry) = real_coords(k, 0, y);
            xlist2.push_back(ry);
            cd val = grid[center[0]][center[1]+y][k];
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

vector<double> dislocAmplitudes;

void step_walk(int step = -1) {
    std::cerr << "Begin step " << step << std::endl;
    grid = applyCoins(grid, U);
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, U*C0*U.adjoint());
        grid = shift(grid);
    }
    grid = applyCoins(grid, U.adjoint());
    double sumDisloc = 0;
    for(int y = 0; y <= ymax; y++)
        for(int k = 0; k < 3; k++)
            sumDisloc += sqmodulus(grid[center[0]][y+center[1]][k]);
    for(int y = 0; y <= ymax && 3*y+1 <= xmax; y++)
        for(int k = 0; k < 3; k++)
            sumDisloc += sqmodulus(grid[center[0]+3*y+1][center[1]+y][k]);
    dislocAmplitudes.push_back(sumDisloc);
    std::cerr << "Total amplitude: " << sumAmplitudes() << "\n";
    std::cerr << "End step " << step << std::endl;
}

void plotDislocAmplitude() {
    std::cerr << "Plotting dislocation amplitude" << std::endl;
    plt::plot(dislocAmplitudes);
    plt::save(prefix + "/" + prefix + "_disloc.png");
    plt::clf();
    plt::close();
}

void print_params() {
    std::ofstream ostream(prefix + "/settings.txt");
    ostream << "num_steps = " << num_steps << "\n";
    ostream << "initial state: " << initialStateName[initialState] << "\n";
    if(plotCone)
        ostream << "Cone coordinates\n";
    else
        ostream << "Unfolded cone\n";
    ostream << "xmin = " << xmin << "\n";
    ostream << "xmax = " << xmax << "\n";
    ostream << "ymin = " << ymin << "\n";
    ostream << "ymax = " << ymax << "\n";
    ostream << "U=\n" << U << "\n";
    ostream << "C0=\n" << C0 << std::endl;
    ostream.close();
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
        "simul_cone_newdisloc_" << initialStateName[initialState];
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
    // A square
    if(initialState == 0) {
        for(int x = xmin/5; x <= xmax/5; x++)
            for(int y = ymin/5; y <= ymax/5; y++)
                for(int k = 0; k < 3; k++)
                    if(x <= 0 || y <= (x-1)/3)
                        grid[center[0]+x][center[1]+y][k] = 1;
        normalizeGrid();
    }
    // Shifted
    else if(initialState == 1) {
        for(int k = 0; k < 3; k++)
            grid[center[0]][center[1]+ymax/2-1][k]=1/sqrt(3);
    }
    // Shifted (in another way)
    else if(initialState == 2) {
        for(int k = 0; k < 3; k++)
            grid[center[0]+xmin/4+1][center[1]+ymin/4+1][k]=1/sqrt(3);
    }
    // Centered
    else if(initialState == 3) {
        vector<vector<int> > centercoords = {{-1,-1},{-1,0},{0,-1},{0,0},{1,-1},{1,0}};
        for(vector<int> coord : centercoords)
            for(int k = 0; k < 3; k++)
                grid[center[0]+coord[0]][center[1]+coord[1]][k]=1/sqrt(3*centercoords.size());
    }
    // Shifted, both sides of the line
    else if(initialState == 4) {
        for(int k=0; k<3; k++) {
            grid[center[0]+10][center[1]+3][k] = 1/sqrt(6);
            grid[center[0]][center[1]+6][k] = 1/sqrt(6);
        }
    }
    // Everywhere, uniformly
    else if(initialState == 5) {
        for(int x = xmin; x <= xmax; x++)
            for(int y = ymin; y <= ymax; y++)
                if(x<=0 || y <= (x-1)/3)
                    for(int k = 0; k < 3; k++)
                        grid[center[0]+x][center[1]+y][k] = 1;
        normalizeGrid();
    }
    plot(0);
    for(int i = 0; i < num_steps; i++) {
        step_walk(i);
        if((i+1)%10 == 0)
            plot(i+1);
    }
    plotDislocAmplitude();

}
