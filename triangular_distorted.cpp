// Compile command: g++ -Wall -Wextra -std=c++17 -O2 -pthread -I/usr/include/eigen3 -I/usr/include/python3.8 -o triangular_distorted triangular_distorted.cpp -lpython3.8
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
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
#include <Python.h>
#include "matplotlibcpp.h"
#include "tinycolormap.hpp"


namespace plt = matplotlibcpp;
using Eigen::MatrixXd;
using Eigen::Matrix2cd;
using Eigen::Vector2cd;
using Eigen::Vector3f;
using std::vector;
using std::sqrt;
using std::cos;
using std::acos;
using std::sin;
using namespace std::complex_literals;

typedef std::complex<double> cd;
typedef vector<vector<vector<cd>>> grid_t;
typedef std::function<Matrix2cd(int, double, double)> gen_coin_t;

// Settings here: epsilon and size (the factor before /sqrt(eps))
// Other settings: 
double eps = 0.01;
int num_steps = 100000;
int deformation = 0;
int initialState = 0;
vector<std::string> initialStateName = {"center", "vertical", "center-side0", "center-side1", "center-side2", "shift-ur", "shift-ur3", "sinx", "siny"};
double dy = sqrt(3);
bool showVectField = false; // Set to true to show the vector field
bool forceSphere = false;
int xspan = (int)(10/sqrt(eps));
int yspan = (int)(xspan/dy);
int ntriangles_x = 2*xspan;
int ntriangles_y = 2*yspan;
int center[2] = {xspan, yspan};

std::time_t now = time(0);
std::tm *ltm = localtime(&now);

std::string prefix;

grid_t zerogrid() {
    return vector<vector<vector<cd>>>(ntriangles_x, vector<vector<cd>> (ntriangles_y, vector<cd> (3, 0. + 0i)));
}

grid_t grid = zerogrid();

double sumAmplitudes() {
    double ret = 0.0;
    for(int x = -xspan; x < xspan; x++)
        for(int y = -yspan; y < yspan; y++)
            for(int side = 0; side < 3; side++) {
                cd val = grid[x+center[0]][y+center[1]][side];
                ret += std::real(val*std::conj(val));
            }
    return ret;
}

void normalizeGrid() {
    double target = sumAmplitudes();
    double mul = 1/sqrt(target);
    if(target == 0)
        return;
    for(int x = -xspan; x < xspan; x++)
        for(int y = -yspan; y < yspan; y++)
            for(int side = 0; side < 3; side++)
                grid[x+center[0]][y+center[1]][side] *= mul;
}

Matrix2cd H, Q;

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

vector<std::string> deform_name = {"ident", "conic", "3fold", "3fold-x", "3fold-y", "sphere-singul", "sphere-nosingul", "cone", "zeroy"};

Matrix2cd lamb(double rx, double ry) {
    Matrix2cd ret;
    // Identité
    if(deformation == 0) {
        ret <<
            1, 0,
            0, 1;
    }
    // Conique : pas de moyen simple d'éviter la singularité
    else if(deformation == 1) {
        if(rx == 0) {
            if(ry >= 0) {
                ret <<
                    -1, 0,
                    0, 1/sqrt(1+4*ry*ry);
            }
            else {
                ret <<
                    1, 0,
                    0, 1/sqrt(1+4*ry*ry);
            }
        }
        else {
            ret <<
                -sign(rx)*ry/rx/sqrt(1+sq(ry/rx)),
                sign(rx)/sqrt(1+sq(2*rx+2*ry*ry/rx)+sq(ry/rx)),
                sign(rx)/sqrt(1+sq(ry/rx)),
                sign(rx)*ry/rx/sqrt(1+sq(2*rx+2*ry*ry/rx)+sq(ry/rx));
        }
    }
    else if(deformation == 2) { // 10-fold expansion of space
        ret <<
            3, 0,
            0, 3;
    }
    else if(deformation == 3) { // 3-fold expansion of space in x direction
        ret <<
            3, 0,
            0, 1;
    }
    else if(deformation == 4) { // 3-fold expansion of space in y direction
        ret <<
            1, 0,
            0, 3;
    }
    // Sphérique
    else if(deformation == 5) { // Sphérique, coordonnées tournantes (singularité en (0,0))
        if(rx == 0 && ry == 0) {
            ret <<
                1, 0,
                0, 1;
        }
        else {
            Vector2cd er, retheta;
            double norm = sqrt(rx*rx+ry*ry);
            double phi = acos(1-4/(norm*norm/2+2));
            er << rx/norm, ry/norm;
            retheta << -ry, rx;
            Vector2cd vect1 = 1/sin(phi)*retheta;
            Vector2cd vect2 = sq(norm*norm/2+2)*sin(phi)/(4*norm)*er;
            ret <<
                vect1(0), vect1(1),
                vect2(0), vect2(1);
        }
    }
    else if(deformation == 6) { // Sphère, sans singularité
        double den = sq(rx*rx+ry*ry+4);
        Vector3f partialx(
            4*(-rx*rx+ry*ry+4)/den,
            -8*rx*ry/den,
            16*rx/den
        );
        Vector3f partialy(
            -8*rx*ry/den,
            4*(rx*rx-ry*ry+4)/den,
            16*ry/den
        );
        ret <<
            1/partialx.norm(), 0,
            0, 1/partialy.norm();
    }
    else if(deformation == 7) {
        double xi = sqrt(rx*rx+ry*ry);
        if(xi <= 1e-10)
            return Matrix2cd::Identity();
        double phi = atan2(ry, rx);
        const double a = 1.0, c = 1.0;
        Vector3f partial_xi(
            cos(phi),
            sin(phi),
            c*xi/a/a/sqrt(1+sq(xi)/sq(a)));
        Vector3f partial_phi(
            -sin(phi),
            cos(phi),
            0
        );
        Vector2cd exi(cos(phi), sin(phi));
        Vector2cd ephi(-sin(phi), cos(phi));
        Vector2cd vect1 = exi/partial_xi.norm(), vect2 = ephi/partial_phi.norm();
        ret <<
            vect1(0), vect2(0),
            vect1(1), vect2(1);
    }
    else if(deformation == 8) {
        ret <<
            1, 0,
            0, 0;
    }
    else
        throw std::runtime_error("Invalid deformation");
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
    for(int x = -xspan; x < xspan; x++) {
        for(int y = -yspan; y < yspan; y++) {
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
            ycoord = y*dy;
        }
        else if(iside == 1) {
            xcoord = x+dec;
            ycoord = y*dy;
        }
        else {
            xcoord = x;
            ycoord = (y+dec)*dy;
        }
    }
    else {
        if(iside == 0) {
            xcoord = x+dec;
            ycoord = y*dy;
        }
        else if(iside == 1) {
            xcoord = x-dec;
            ycoord = y*dy;
        }
        else {
            xcoord = x;
            ycoord = (y-dec)*dy;
        }
    }
    return std::make_pair(sqrt(eps)*xcoord, sqrt(eps)*ycoord);
}

const int DELTAS[][2] = {{1,0}, {-1,0}, {0,-1}};
const int NUM_THREADS = 8;

void applyCoinsPartial(grid_t &ngrid, grid_t &grid, gen_coin_t &gen_coin, int xmin, int xmax) {
    for(int x = xmin; x < xmax; x++) {
        for(int y = -yspan; y < yspan; y++) {
            if((x+y)%2)
                continue;
            for(int iside = 0; iside < 3; iside++) {
                cd thisval = grid[x+center[0]][y+center[1]][iside];
                int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
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
    grid_t ngrid = zerogrid();
    if(!multithread) {
        for(int x = -xspan; x < xspan; x++) {
            for(int y = -yspan; y < yspan; y++) {
                if((x+y)%2)
                    continue;
                for(int iside = 0; iside < 3; iside++) {
                    cd thisval = grid[x+center[0]][y+center[1]][iside];
                    int xo = x + DELTAS[iside][0], yo = y + DELTAS[iside][1];
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
            threads[iThread] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(gen_coin), -xspan+iThread*delta_x, -xspan+(iThread+1)*delta_x);
        }
        threads[NUM_THREADS-1] = std::thread(applyCoinsPartial, std::ref(ngrid), std::ref(grid), std::ref(gen_coin), -xspan+(NUM_THREADS-1)*delta_x, xspan);
        for(int iThread = 0; iThread < NUM_THREADS; iThread++)
            threads[iThread].join();
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

vector<double> sphereCoords(double x, double y) {
    double den = .5*(x*x+y*y)+2;
    return {2*x/den, 2*y/den, 1-4/den};
}

void plotSphere(int iGrid = -1) {
    vector<vector<double>> xgrid(2*xspan, vector<double>(2*yspan, 0));
    vector<vector<double>> ygrid(2*xspan, vector<double>(2*yspan, 0));
    vector<vector<double>> zgrid(2*xspan, vector<double>(2*yspan, 0));
    vector<vector<double>> colgrid(2*xspan, vector<double>(2*yspan, 0));
    vector<double> listcol;
    for(int x = -xspan; x < xspan; x++)
        for(int y = -yspan; y < yspan; y++) {
            double rx, ry;
            std::tie(rx, ry) = real_coords(0, x, y, true);
            vector<double> coords = sphereCoords(rx, ry);
            xgrid[x+center[0]][y+center[1]] = coords[0];
            ygrid[x+center[0]][y+center[1]] = coords[1];
            zgrid[x+center[0]][y+center[1]] = coords[2];
            vector<cd> &vals = grid[x+center[0]][y+center[1]];
            double sum = 0.0;
            for(int i = 0; i < 3; i++)
                sum += std::real(vals[i]*std::conj(vals[i]));
            //sum /= approxArea(rx, ry);
            colgrid[x+center[0]][y+center[1]] = sum;
            listcol.push_back(sum);
        }
    std::sort(listcol.rbegin(), listcol.rend());
    double maxcol = (listcol[0]+listcol[1])/2;
    if(maxcol == 0.0)
        maxcol = 1.0;
    maxcol *= .6;
    vector<vector<vector<double>>> facecolors(2*xspan, vector<vector<double>> (2*yspan, vector<double> (4,1.0)));
    for(int x = -xspan; x < xspan; x++)
        for(int y = -yspan; y < yspan; y++) {
            double val = std::max(0.0, std::min(1.0, colgrid[x+center[0]][y+center[1]]/maxcol));
            tinycolormap::Color col = tinycolormap::GetViridisColor(1-val);
            for(int i = 0; i < 3; i++)
                facecolors[x+center[0]][y+center[1]][i] = col.data[i];
        }
    PyObject* fig = plt::plot_surface(xgrid, ygrid, zgrid, {}, facecolors, 1000, 1000, {-45.0, 30.0});
    std::ostringstream filename;
    filename << prefix << "_sphere_" << iGrid << ".png";
    plt::save(prefix + "/" + filename.str());
    plt::cla();
    plt::clf();
    plt::close(fig);
    Py_DECREF(fig);
}

void plot(int iGrid = -1) {
    PyObject* fig = plt::figure_size(1000,1000);
    /*plt::xlim(-xspan*sqrt(eps), xspan*sqrt(eps));
    plt::ylim(-yspan*sqrt(eps)*dy, yspan*sqrt(eps)*dy);*/
    plt::set_aspect_equal();
    vector<vector<double>> imgrid(2*yspan, vector<double>(2*xspan, 0.0));
    vector<double> listvals;
    for(int y = -yspan; y < yspan; y++) {
        for(int x = -xspan; x < xspan; x++) {
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
    double minx = sqrt(eps)*(-xspan-.5);
    double maxx = sqrt(eps)*(xspan-.5);
    double miny = sqrt(eps)*dy*(-yspan-.5);
    double maxy = sqrt(eps)*dy*(yspan-.5);
    plt::imshow(imgrid, {minx, maxx, miny, maxy}, {{"origin", "lower"}, {"cmap", "viridis_r"}, {"vmin", "0.0"}, {"vmax", std::to_string(maxi)}});
    if(showVectField)
        plotVectorField(-xspan*sqrt(eps), (xspan-1)*sqrt(eps), -yspan*sqrt(eps)*dy, (yspan-1)*sqrt(eps)*dy, 20);
    std::ostringstream filename;
    filename << prefix << "_" << iGrid << ".png";
    plt::save(prefix + "/" + filename.str());
    plt::cla();
    plt::clf();
    plt::close(fig);
    Py_DECREF(fig);
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
            Matrix2cd ret = gen_U(modulo(i-1,3), rx, ry);
            return ret;
        });
    }
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_U);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_Ustar(modulo(i-1,3), rx, ry);
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
            Matrix2cd ret = gen_Ubis(modulo(i-1,3), rx, ry);
            return ret;
        });
    }
    for(int j = 0; j < 3; j++) {
        grid = applyCoins(grid, gen_Ubis);
        grid = shift(grid);
        grid = applyCoins(grid, [](int i, double rx, double ry) {
            Matrix2cd ret = gen_Ubisstar(modulo(i-1,3), rx, ry);
            return ret;
        });
    }
    grid = applyCoins(grid, [](int i, double rx, double ry) {
        Matrix2cd ret = Q.adjoint();
        return ret;
    });
    std::cerr << "Total amplitude: " << sumAmplitudes() << "\n";
    std::cerr << "End step " << step << std::endl;
}

void print_params() {
    std::ofstream ostream(prefix + "/settings.txt");
    ostream << "num_steps = " << num_steps << "\n";
    ostream << "eps = " << eps << "\n";
    ostream << "xspan = " << xspan << "\n";
    ostream << "yspan = " << yspan << "\n";
    ostream << "deformation: " << deform_name[deformation] << "\n";
    ostream << "initial state: " << initialStateName[initialState] << "\n";
    /*ostream << "Deformation matrix at (1,1):\n" << lamb(1,1) << "\n";
    for(int i = 0; i < 6; i++) {
        ostream << "U" << i << "(1,1):\n" << gen_U(i,1,1) << "\n";
    }
    ostream << "H=\n" << H << "\n";
    ostream << "Q=\n" << Q << std::endl;*/
    ostream.close();
}

void listDeformations() {
    for(size_t i = 0; i < deform_name.size(); i++) {
        std::cerr << "- " << i << ": " << deform_name[i] << "\n";
    }
}

void listInitialStates() {
    for(size_t i = 0; i < initialStateName.size(); i++)
        std::cerr << "- " << i << ": " << initialStateName[i] << "\n";
}

int main(int argc, char **argv)
{
    if(argc <= 2) {
        std::cerr << "Usage: " << std::string(argv[0]) << " <deformation> <initial state> [force sphere].\n";
        std::cerr << "Deformations:\n";
        listDeformations();
        std::cerr << "Initial states:\n";
        listInitialStates();
        std::cerr << "Put any non-empty third argument to force plotting on a sphere.\n";
        return 1;
    }
    deformation = std::atoi(argv[1]);
    if(deformation < 0 || deformation >= (int)deform_name.size()) {
        std::cerr << "Invalid deformation " << deformation << ". List of deformations:\n";
        listDeformations();
        return 2;
    }
    initialState = std::atoi(argv[2]);
    if(initialState < 0 || initialState >= (int)initialStateName.size()) {
        std::cerr << "Invalid initial state " << initialState << ". List of initial states:\n";
        listInitialStates();
        return 3;
    }
    forceSphere = argc >= 4;
    std::cout << "Deformation selected: " << deformation << " " << deform_name[deformation] << std::endl;
    std::ostringstream str;
    str <<
        "simul_distorted_" << deform_name[deformation] << "_" << initialStateName[initialState] << "_" <<
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
    // Vertical
    if(initialState == 1)
        for(int y = -yspan; y < yspan; y++)
            for(int k = 0; k < 3; k++)
                grid[center[0]][y+center[1]][k] = 1;
    // Centered
    else if(initialState == 0)
        for(int k = 0; k < 3; k++)
            grid[center[0]][center[1]][k]=1/sqrt(3);
    // Centered, side 0
    else if(initialState == 2)
        grid[center[0]][center[1]][0] = 1;
    // Centered, side 1
    else if(initialState == 3)
        grid[center[0]][center[1]][1] = 1;
    // Centered, side 2
    else if(initialState == 4)
        grid[center[0]][center[1]][2] = 1;
    // Shifted up-right at half
    else if(initialState == 5)
        for(int k = 0; k < 3; k++)
            grid[center[0]+xspan/2][center[1]+yspan/2][k] = 1/sqrt(3);
    // Shifted up-right at third
    else if(initialState == 6)
        for(int k = 0; k < 3; k++)
            grid[center[0]+xspan/3][center[1]+yspan/3][k] = 1/sqrt(3);
    // Sine, x dimension
    else if(initialState == 7) {
        const double rxmin = (-xspan-.5)*sqrt(eps);
        const double rxmax = (xspan-.5)*sqrt(eps);
        for(int x = -xspan; x < xspan; x++)
            for(int y = -yspan; y < yspan; y++)
                for(int k = 0; k < 3; k++) {
                    double rx, ry;
                    std::tie(rx, ry) = real_coords(k,x,y);
                    grid[center[0]+x][center[1]+y][k] = sin(2*M_PI*(float)(rx-rxmin)/(rxmax-rxmin));
                }
        normalizeGrid();
    }
    // Sine, y dimension
    else if(initialState == 8) {
        const double rymin = (-yspan-.5)*sqrt(eps);
        const double rymax = (yspan-.5)*sqrt(eps);
        for(int x = -xspan; x < xspan; x++)
            for(int y = -yspan; y < yspan; y++)
                for(int k = 0; k < 3; k++) {
                    double rx, ry;
                    std::tie(rx, ry) = real_coords(k,x,y);
                    grid[center[0]+x][center[1]+y][k] = sin(2*M_PI*(float)(ry-rymin)/(rymax-rymin));
                }
        normalizeGrid();
    }
    plot(0);
    bool isSphere = deformation == 5 || deformation == 6;
    if(isSphere || forceSphere)
        plotSphere(0);
    for(int i = 0; i < num_steps; i++) {
        step_walk(i);
        if((i+1)%10 == 0) {
            plot(i+1);
            if(isSphere || forceSphere)
                plotSphere(i+1);
        }
    }
}
