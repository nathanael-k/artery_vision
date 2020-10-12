//
// Created by nate on 19.09.20.
//

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"

#include <iostream>
#include <chrono>

#include <camera.h>
#include <arteryNet.h>
#include <imageData.h>
#include <list>

imageData data1("../data/renders/easy_flow_3/", 0),
          data2("../data/renders/easy_flow_3/", 1);

void displayVisual( int, void* );
void changeVisual( int pos, void* );
void buildGraph(imageData& data1, imageData& data2);

void debugWaitShow() {
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
    cv::waitKey(0);
}

int main( int argc, char** argv )
{
    if( data1.source.empty() ||  data2.source.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }


    // create windows
    cv::namedWindow( "Control");
    cv::namedWindow( "Cam1 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Visual", cv::WINDOW_AUTOSIZE);
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);

    int what = 0;

    cv::createTrackbar( "Frame:", "Control", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Visualisation Source:", "Control", &what, 6, changeVisual);

    // prepare skeletons
    data1.SkeletonizeAll();
    data2.SkeletonizeAll();

    // prepare endpoints
    data1.EndpointsAll();
    data2.EndpointsAll();

    buildGraph(data1, data2);

    // input starting location to process (will be the position of catheter tip later on)
    // register mouse callback
    //cv::setMouseCallback("Cam1 Source", onMouse1);
    //cv::setMouseCallback("Cam2 Source", onMouse2);

    cv::waitKey(0);
    return 0;
}

namespace point {


// to classify type of special points
    enum class Type {
        endpoint, //1 line
        path, //2 lines
        junction, //3 lines
        pseudo_junctionA,
        pseudo_junctionB //also 3 lines, but not a real junction
    };

    enum class Status {
        fresh,
        frontier,
        old
    };

}

struct specialPoint {
    // index in the list of all located points
    int index;
    // number in the tree hierarchy
    int number = -1;

    point::Type type = point::Type::endpoint;
    point::Status status = point::Status::fresh;

    Vector2d pos_1;
    Vector2d pos_2;
    double distance;

    bool origin_1 = false;
    bool origin_2 = false;

    bool found_1 = false;
    bool found_2 = false;

    int lines_1[3];
    int lines_2[3];

    int n_lines_1 = 0;
    int n_lines_2 = 0;

    arteryNode* node = nullptr;
};

bool trace(imageData& data1, imageData& data2, int index, specialPoint& point,
          std::vector<specialPoint>& specialPoints);

int pointCloseTo(const cv::Mat& img, cv::Mat& vis, uchar value, Vector2d& position, int range=10);

struct Edge{
    // indexA < indexB
    int indexA_ = -1;
    int indexB_ = -1;

    bool replaced_from_old_frame = false;
    bool found_in_old_frame = false;
    bool found_in_next_frame = false;
    double length = -1;

    bool valid() {
        //assert (indexA_ <= indexB_);
        return (indexA_ >= 0 && indexB_ >= 0);
    }

    int connects_to(int me) {
        assert(valid());
        int a = indexA_;
        int b = indexB_;
        if (me == a) {
            return b;
        }
        if (me == b) {
            return a;
        }
        assert(false);
    }

    void add_index(int index) {
        // already added
        if (index == indexA_ || index == indexB_)
            return;

        if (indexA_ == -1)
            indexA_ = index;
        else
        {
            assert(indexB_ == -1);
            if (indexA_ < index)
                indexB_ = index;
            else {
                indexB_ = indexA_;
                indexA_ = index;
            }
        }
    }

    bool operator== (const Edge& rhs) const {
        return indexA_ == rhs.indexA_ && indexB_ == rhs.indexB_;
    }
};

// checks which value appears the most and returns it (suppressing 0)
int countPixels(const cv::Mat& img, std::vector<int>& tally) {
    for (int i = 0; i<img.cols; i++)
        for (int j = 0; j<img.rows; j++) {
            int value = img.at<uchar>(j,i);
            tally[value]++;
        }

    int max;
    int amount = 0;

    for (int i = 1; i < tally.size(); i++)
    {
        if (tally[i] > amount)
        {
            amount = tally[i];
            max = i;
        }
    }

    return max;
}

// paints all pixels with a value in list on reference as black in destination
void suppress(const cv::Mat& ref, cv::Mat& dest, std::vector<int>& list) {
    if (list.empty())
        return;
    int max = *std::max_element(list.begin(), list.end());
    std::vector<bool> suppr(max+1, false);
    for (auto& i : list) {
        suppr[i] = true;
    }
    for (int i = 0; i<ref.cols; i++)
        for (int j = 0; j<ref.rows; j++) {
            int value = ref.at<uchar>(j,i);
            if (value <= max && suppr[value] == true) {
                dest.at<uchar>(j,i) = 0;
            }
        }
}

int processEdges(const std::vector<Edge>& edges, std::vector<specialPoint>& points, int index, int& added_points);

void matchPoints(std::list<Vector2d>& list1, std::list<Vector2d>& list2,
                 const Camera& cam1, const Camera& cam2, arteryGraph& graph,
                 std::vector<specialPoint>& points, point::Type type, double distance_limit = 0.5) {

    // find best match and remove, until the match is over the threshold
    while(true) {
        double shortest_distance = std::numeric_limits<double>::max();
        Vector3d best_position;
        auto best_pos1 = list1.begin();
        auto best_pos2 = list2.begin();;
        for (auto it1 = list1.begin(); it1 != list1.end(); it1++) {
            for (auto it2 = list2.begin(); it2 != list2.end(); it2++) {
                Vector3d position;
                double distance = Camera::intersect(cam1.origin, cam1.ray(*it1).normalized(),
                                                    cam2.origin, cam2.ray(*it2).normalized(), position);
                if (distance < shortest_distance) {
                    shortest_distance = distance;
                    best_position = position;
                    best_pos1 = it1;
                    best_pos2 = it2;
                }
            }
        }
        if (shortest_distance > distance_limit)
            return;
        else {

            // add to list of Points
            specialPoint point;
            point.found_1 = point.found_2 = true;
            point.origin_1 = point.origin_2 = true;
            point.pos_1 = *best_pos1;
            point.pos_2 = *best_pos2;
            point.index = points.size();
            point.distance = shortest_distance;
            point.type = type;
            points.push_back(point);
            //remove already matched
            list1.erase(best_pos1);
            list2.erase(best_pos2);
        }
    }
}

// builds a graph from matching image sequences
// 1) find endpoints or junctions in both frames at same time
// 2) greedily match them to find 3d positions, add those to the graph
// 3) also create nodes for (and collect) unmatched points
// 4) mark all points in the buffer to find them fast
// 5) trace image (on skeleton) beginning at matched endpoints, until you find a marked point in the buffer
// 6) if that point is already matched, connect in the graph and continue
// 7) unmatched points can be easily matched because you know locally where you are (thanks to tracing) -> match them and add them to the graph
// when all are matched, correlate those matches to next endpoints layer / skeleton
// junctions are expected to stay, however if a path appears again its now a (final) endpoint
// for a match: delete the respective pixel in the endpoints layer
// paint all known matches in the buffer layer of the next time step (at correct location)
void buildGraph(imageData& data1, imageData& data2) {

    arteryGraph& graph = data1.graph;
    std::vector<specialPoint> locatedPoints;
    std::vector<Edge> former_edges_in_1;
    std::vector<Edge> former_edges_in_2;
    int finishedPoints = 0;
    int addedPoints = 0;

    // prepare
    cv::bitwise_not(data1.endpoints[0], data1.buffer[0]);
    cv::bitwise_not(data2.endpoints[0], data2.buffer[0]);

    // loop over all images in the sequence
    for (int index = 0; index<data1.size; index++) {

        // find NEW special points (endpoints or junctions) in both frames
        std::list<Vector2d> ends1;
        std::list<Vector2d> ends2;
        std::list<Vector2d> junctions1;
        std::list<Vector2d> junctions2;
        //std::list<Vector2d> pseudo_junctions1;
        //std::list<Vector2d> pseudo_junctions2;
        //std::list<Vector2d> located_pseudo_junctions;

        // iterate over whole picture, put special pixel into vector to process them later
        for (int j = 0; j < data1.endpoints[index].rows; j++) {

            // for fast matrix access
            uchar* pixel1 = data1.endpoints[index].ptr<uchar>(j);
            uchar* pixel2 = data2.endpoints[index].ptr<uchar>(j);

            for (int k = 0; k<data1.endpoints[index].cols; k++) {
                Vector2d place = Vector2d(k,j);
                if (*pixel1 == 255) // end
                    ends1.emplace_back(place);
                if (*pixel1 == 128) {
                    //auto dummy = place;
                    //if (index > 0 && pointCloseTo(data1.skeleton[index-1], data1.visualisation[index-1], 255, dummy, 30) > -1)
                    //    pseudo_junctions1.emplace_back(place);
                    //else
                        junctions1.emplace_back(place);
                }
                if (*pixel2 == 255)
                    ends2.emplace_back(place);
                if (*pixel2 == 128) {
                    //auto dummy = place;
                    //if (index > 0 && pointCloseTo(data2.skeleton[index-1], data2.visualisation[index-1], 255, dummy, 30) > -1)
                    //    pseudo_junctions2.emplace_back(place);
                    //else
                        junctions2.emplace_back(place);
                }
                pixel1++;
                pixel2++;
            }
        }

        matchPoints(junctions1, junctions2, data1.cam, data2.cam,graph,locatedPoints, point::Type::junction);

        matchPoints(ends1, ends2, data1.cam, data2.cam,graph,locatedPoints, point::Type::endpoint);

        // also match still unmatched ends to junctions
        matchPoints(ends1, junctions2, data1.cam, data2.cam,graph,locatedPoints, point::Type::pseudo_junctionB);
        matchPoints(junctions1, ends2, data1.cam, data2.cam,graph,locatedPoints, point::Type::pseudo_junctionA);

        //matchPoints(ends1, pseudo_junctions2, data1.cam, data2.cam,graph,locatedPoints, point::Type::pseudo_junctionB, 3);
        //matchPoints(pseudo_junctions1, ends2, data1.cam, data2.cam,graph,locatedPoints, point::Type::pseudo_junctionA, 3);

        // 3) also create nodes for (and collect) unmatched points
        for (auto& junction : junctions1)
        {
            specialPoint point;
            point.found_1 = true;
            point.origin_1 = true;
            point.pos_1 = junction;
            point.index = locatedPoints.size();
            point.type = point::Type::junction;
            locatedPoints.push_back(point);
        }

        for (auto& junction : junctions2)
        {
            specialPoint point;
            point.found_2 = true;
            point.origin_2 = true;
            point.pos_2 = junction;
            point.index = locatedPoints.size();
            point.type = point::Type::junction;
            locatedPoints.push_back(point);
        }
/*
        for (auto& ps_junction : pseudo_junctions1)
        {
            specialPoint point;
            point.found_1 = true;
            point.origin_1 = true;
            point.pos_1 = ps_junction;
            point.index = locatedPoints.size();
            point.type = point::Type::pseudo_junctionA;
            locatedPoints.push_back(point);
        }

        for (auto& ps_junction : pseudo_junctions2)
        {
            specialPoint point;
            point.found_2 = true;
            point.origin_2 = true;
            point.pos_2 = ps_junction;
            point.index = locatedPoints.size();
            point.type = point::Type::pseudo_junctionB;
            locatedPoints.push_back(point);
        }
*/


        for (auto& end : ends1)
        {
            specialPoint point;
            point.found_1 = true;
            point.origin_1 = true;
            point.pos_1 = end;
            point.index = locatedPoints.size();
            locatedPoints.push_back(point);
        }
        for (auto& end : ends2)
        {
            specialPoint point;
            point.found_2 = true;
            point.origin_2 = true;
            point.pos_2 = end;
            point.index = locatedPoints.size();
            locatedPoints.push_back(point);
        }

        // 4) mark all points in the current buffer to find them fast
        // also disconnect skeleton at the points so that we can find the components
        for (auto& point : locatedPoints) {

            if (point.status == point::Status::fresh) {
                if (point.found_1) {
                    data1.buffer[index].at<uchar>(point.pos_1.y(), point.pos_1.x()) = point.index;

                    if (point.origin_1)
                        data1.skeleton[index].at<uchar>(point.pos_1.y(), point.pos_1.x()) = 0;
                }

                if (point.found_2) {
                    data2.buffer[index].at<uchar>(point.pos_2.y(), point.pos_2.x()) = point.index;

                    if (point.origin_2)
                        data2.skeleton[index].at<uchar>(point.pos_2.y(), point.pos_2.x()) = 0;
                }
            }
        }

        // mark/find connected components of skeleton
        // TODO: remove and put to preprocessing (how to handle edge_count?)
        int edge_count_1 = cv::connectedComponents(data1.skeleton[index], data1.components[index], 8, CV_16U) -1;
        int edge_count_2 = cv::connectedComponents(data2.skeleton[index], data2.components[index], 8, CV_16U) -1;
        data1.components[index].convertTo(data1.components[index], CV_8U);
        data2.components[index].convertTo(data2.components[index], CV_8U);

        // collect edges
        std::vector<Edge> edges_in_1(edge_count_1);
        std::vector<Edge> edges_in_2(edge_count_2);

        for(auto& point : locatedPoints) {
            // find out what edge this point is connected to
            // if (point.index == 18) debugWaitShow();
            int max_connections = 1;
            if (point.type == point::Type::path)     max_connections = 2;
            if (point.type == point::Type::junction || point.type == point::Type::pseudo_junctionA || point.type == point::Type::pseudo_junctionB)
                max_connections = 3;
            int found_connections_cam1 = 0;
            int found_connections_cam2 = 0;

            // work on camera 1 first
            if (point.origin_1)
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    if (i != 0 || j != 0) {
                        Vector2d trialPoint = point.pos_1 + Vector2d(i, j);
                        int edge = data1.components[index].at<uchar>(trialPoint.y(), trialPoint.x());
                        if (edge != 0) {
                            point.lines_1[found_connections_cam1] = edge - 1;
                            found_connections_cam1++;
                            edges_in_1[edge-1].add_index(point.index);
                        }
                    }
            // work on camera 2
            if (point.origin_2)
                for (int i = -1; i <= 1; i++)
                    for (int j = -1; j <= 1; j++)
                        if (i != 0 || j != 0) {
                            Vector2d trialPoint = point.pos_2 + Vector2d(i, j);
                            int edge = data2.components[index].at<uchar>(trialPoint.y(), trialPoint.x());
                            if (edge != 0) {
                                point.lines_2[found_connections_cam2] = edge - 1;
                                found_connections_cam2++;
                                edges_in_2[edge-1].add_index(point.index);
                            }
                        }
            // this is only relevant for new points!
            // if (point.status == point::Status::fresh)
                assert(found_connections_cam1 <= max_connections &&
                    found_connections_cam2 <= max_connections);
                        point.n_lines_1 = found_connections_cam1;
                        point.n_lines_2 = found_connections_cam2;
        }



        // check all edges and see how long they are
        for (auto& edge : edges_in_1) {
            assert(edge.valid());
            edge.length = (locatedPoints[edge.indexA_].pos_1 - locatedPoints[edge.indexB_].pos_1).norm();
        }
        for (auto& edge : edges_in_2) {
            assert(edge.valid());
            edge.length = (locatedPoints[edge.indexA_].pos_2 - locatedPoints[edge.indexB_].pos_2).norm();
        }


        // only first round, add root
        if (index == 0) {
            // right now we need at least one matched node, this will be the root node
            auto& point = locatedPoints[0];
            assert(point.found_1 && point.found_2);
            point.number = addedPoints;
            Vector3d position;
            double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(point.pos_1).normalized(),
                                                data2.cam.origin, data2.cam.ray(point.pos_2).normalized(), position);
            point.node = new arteryNode(graph,position);
            addedPoints++;
            point.status = point::Status::frontier;
            graph.root = locatedPoints[0].node;
            graph.size = 1;
            finishedPoints = 1;
        }

        // generate frontier
        std::vector<specialPoint> frontier;
        for (auto& point : locatedPoints)
            if (point.status == point::Status::frontier)
                frontier.push_back(point);

        if (index == 0)
            frontier.push_back(locatedPoints[0]);

        //assert(frontier.size() > 0);

        // TODO: add clever edge processing here! (not needed right now)
        if (index != 0) {
            // mark all edges that already existed before as "old" (we then won't process these again)
            for (auto &edge : edges_in_1) {
                for (auto &former_edge : former_edges_in_1) {
                    if (edge == former_edge) {
                        edge.found_in_old_frame = true;
                        former_edge.found_in_next_frame = true;
                        break;
                    }
                }
            }
            for (auto &edge : edges_in_2) {
                for (auto &former_edge : former_edges_in_2) {
                    if (edge == former_edge) {
                        edge.found_in_old_frame = true;
                        former_edge.found_in_next_frame = true;
                        break;
                    }
                }
            }}

        // we can check each new junctions what it connects to
        for (auto& point : locatedPoints) {
            if (point.status == point::Status::fresh && point.type == point::Type::junction) {

                // TODO: if two junctions see each other, they must be pseudo junctions

                int old_points[3];
                int new_points[3];

                int found_old_points = 0;
                int found_new_points = 0;

                int* lines;
                if (point.origin_1) {
                    lines = point.lines_1;
                    assert (point.n_lines_1 == 3);
                }
                else {
                    lines = point.lines_2;
                    assert (point.n_lines_2 == 3);
                }

                int neighbours[3];
                for (int i = 0; i<3; i++) {
                    if (point.origin_1) neighbours[i] = edges_in_1[lines[i]].connects_to(point.index);
                    else neighbours[i] = edges_in_2[lines[i]].connects_to(point.index);
                }

                for (int i = 0; i<3; i++) {
                    if (locatedPoints[neighbours[i]].status == point::Status::old) {
                        old_points[found_old_points] = neighbours[i];
                        found_old_points++;

                        // mark that edge as found
                        Edge found; found.add_index(point.index); found.add_index(neighbours[i]);
                        if (point.origin_1) {
                            for (auto &former_edge : former_edges_in_1) {
                                if (found == former_edge) {
                                    former_edge.found_in_next_frame = true;
                                    break;
                                }
                            }
                            edges_in_1[lines[i]].found_in_old_frame = true;
                        }
                        if (point.origin_2) {
                            for (auto &former_edge : former_edges_in_2) {
                                if (found == former_edge) {
                                    former_edge.found_in_next_frame = true;
                                    break;
                                }
                            }
                            edges_in_2[lines[i]].found_in_old_frame = true;
                        }

                    } else {
                        new_points[found_new_points] = lines[i];
                        found_new_points++;
                    }
                }

                //assert(found_new_points <3);
                assert(found_old_points <3);

                // if we connect to two old points, we must be a pseudo junction
                if (found_old_points == 2) {
                    assert(!point.origin_1 || !point.origin_2);
                    if (point.origin_1) point.type = point::Type::pseudo_junctionA;
                    else point.type = point::Type::pseudo_junctionB;
                }
            }
        }

        // TODO: first, connect pseudo junctions to the frontier, then mark that junction also as a frontier
        // TODO: if you have a pseudojunction with only two old connections, and a new one: the closest frontier pseudoJunction is his connection!

        // TODO: all is set to trace with A*: ideally an implementation that will allow reuse
        // returns std::vector<int> nodes - intermediate nodes we pass thru on our way
        // then mark edge as found, and passing nodes edges as "replaced_from_old_frame"
        // then junctions which touch 2 a "replaced from old" edges, they definitely are pseudo junctions
        // with that knowledge we should be able to find out where these pseudo junctions connect to,
        // with priority given to frontier points! then we connect and place pseudo junctions to other pseudo junctions
        // and finally we connect new points to pseudo junctions
        // TODO: then problem will be done, probably!

        // go trough former edges and make sure we find them all
        for (auto& edge: former_edges_in_1) {
            if (!edge.found_in_next_frame) {
                // TODO:
                //debugWaitShow();
            }
        }
        // go trough former edges and make sure we find them all
        for (auto& edge: former_edges_in_2) {
            if (!edge.found_in_next_frame) {
                // TODO:
                //debugWaitShow();
            }
        }

            // suppress (paint black) all pixel that were matched to an old edge
            std::vector<int> suppress_1, suppress_2;
            for (int i = 0; i < edges_in_1.size(); i++) {
                if (edges_in_1[i].found_in_old_frame)
                    suppress_1.push_back(i+1);
            }
            for (int i = 0; i < edges_in_2.size(); i++) {
                if (edges_in_2[i].found_in_old_frame)
                    suppress_2.push_back(i+1);
            }
            data1.skeleton[index].copyTo(data1.new_skeleton[index]);
            data2.skeleton[index].copyTo(data2.new_skeleton[index]);
            suppress(data1.components[index], data1.new_skeleton[index], suppress_1);
            suppress(data2.components[index], data2.new_skeleton[index], suppress_2);



        // first process edges that connect pseudo junctions to the frontier
        for (int i = 0; i < 3; i++)
        {
            std::vector<Edge> edges_to_process_frontier;
            for (auto& edge : edges_in_1)
                if (edge.valid()) {
                    auto &pointA = locatedPoints[edge.indexA_];
                    auto &pointB = locatedPoints[edge.indexB_];
                    if (pointA.status == point::Status::frontier && pointB.type == point::Type::pseudo_junctionA) {
                        edges_to_process_frontier.push_back(edge);
                        pointB.status = point::Status::frontier;
                    }
                }
            for (auto& edge : edges_in_2)
                if (edge.valid()) {
                    auto &pointA = locatedPoints[edge.indexA_];
                    auto &pointB = locatedPoints[edge.indexB_];
                    if (pointA.status == point::Status::frontier && pointB.type == point::Type::pseudo_junctionB) {
                        edges_to_process_frontier.push_back(edge);
                        pointB.status = point::Status::frontier;
                    }
                }
            int edges_left = processEdges(edges_to_process_frontier, locatedPoints, index, addedPoints);
            assert(edges_left == 0);
        }

        // we can start by desperately trying to fix any pseudo junction, since they should always come in pairs.
        // right now we just match any new pseudo junction to the next best pseudo junction that is still open
        for (auto& pointB : locatedPoints) {
            if ((pointB.type == point::Type::pseudo_junctionA || pointB.type == point::Type::pseudo_junctionB)
                && pointB.status == point::Status::fresh) {
                // lets find a already connected partner
                for (auto& pointA : locatedPoints) {
                    if ((pointA.type == point::Type::pseudo_junctionA || pointA.type == point::Type::pseudo_junctionB)
                        && pointA.status == point::Status::frontier) {
                        if (!pointB.found_1) {
                            assert(pointB.found_2);
                            // for debugging, first really actually draw this line in visualisation
                            Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.cam.ray(pointB.pos_2));
                            data1.renderLine(line, index);
                            pointB.pos_1 = double_locate(data1.new_skeleton[index], data1.skeleton[index], data1.components[index], line, pointA.pos_1,
                                                    data2.cam, pointB.pos_2, data1.cam);
                            if (pointB.pos_1 == Vector2d(-1, -1))
                                debugWaitShow();
                            pointB.found_1 = true;
                        }
                        if (!pointB.found_2) {
                            assert(pointB.found_1);
                            // for debugging, first really actually draw this line in visualisation
                            Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.cam.ray(pointB.pos_1));
                            data2.renderLine(line, index);
                            pointB.pos_2 = double_locate(data2.new_skeleton[index], data2.skeleton[index], data2.components[index], line, pointA.pos_2,
                                                    data1.cam, pointB.pos_1, data2.cam);
                            if (pointB.pos_2 == Vector2d(-1, -1))
                                debugWaitShow();
                            pointB.found_2 = true;
                        }
                        assert(!pointB.node);
                        Vector3d position;
                        double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(pointB.pos_1).normalized(),
                                                            data2.cam.origin, data2.cam.ray(pointB.pos_2).normalized(), position);
                        assert (distance < 2);
                        pointB.node = pointA.node->addNode(position);
                        pointB.number = addedPoints;
                        addedPoints++;

                        // these matched junctions are now old
                        pointA.status = point::Status::old;
                        pointB.status = point::Status::old;
                    }

                }
            }
        }

        // if we still have a pseudo junction that is not connected, then we should maybe connect it to the closest frontier?
        // a way to do this would be to locate it as usual, check what edge we hit, remove that edge, and add the relevant edges again?
        for (auto& point : locatedPoints) if(point.type == point::Type::pseudo_junctionA && point.status == point::Status::fresh) {
                Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.cam.ray(point.pos_1));
                point.pos_2 = double_locate(data2.new_skeleton[index], data2.skeleton[index], data2.components[index], line, {data2.cam.resolution/2,data2.cam.resolution/2},
                                        data1.cam, point.pos_1, data2.cam);
                if (point.pos_2 == Vector2d(-1, -1))
                    debugWaitShow();
                point.found_2 = true;

                int edge_2 = data2.components[index].at<uchar>(point.pos_2.y(), point.pos_2.x());
                Edge& edge = edges_in_2[edge_2-1];
                Edge new_1; new_1.add_index(point.index); new_1.add_index(edge.indexB_);
                Edge new_2; new_2.add_index(point.index); new_2.add_index(edge.indexA_);

                edges_in_2[edge_2-1] = new_1;
                edges_in_2.push_back(new_2);
        }

        for (auto& point : locatedPoints) if(point.type == point::Type::pseudo_junctionB && point.status == point::Status::fresh) {
                Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.cam.ray(point.pos_2));
                point.pos_1 = double_locate(data1.new_skeleton[index], data1.skeleton[index], data1.components[index], line, {data1.cam.resolution/2,data1.cam.resolution/2},
                                            data2.cam, point.pos_2, data1.cam);
                if (point.pos_1 == Vector2d(-1, -1))
                    debugWaitShow();
                point.found_1 = true;

                int edge_1 = data1.components[index].at<uchar>(point.pos_1.y(), point.pos_1.x());
                Edge& edge = edges_in_1[edge_1-1];
                Edge new_1; new_1.add_index(point.index); new_1.add_index(edge.indexB_);
                Edge new_2; new_2.add_index(point.index); new_2.add_index(edge.indexA_);

                edges_in_1[edge_1-1] = new_1;
                edges_in_1.push_back(new_2);
            }

        {
            //prioritize edges from pseudo junctions
            std::vector<Edge> edges_to_process_pseudojunctions;
            for (auto& edge : edges_in_1)
                if (edge.valid() && edge.indexA_ < finishedPoints && locatedPoints[edge.indexB_].type == point::Type::pseudo_junctionB)
                    edges_to_process_pseudojunctions.push_back(edge);
            for (auto& edge : edges_in_2)
                if (edge.valid() && edge.indexA_ < finishedPoints && locatedPoints[edge.indexB_].type == point::Type::pseudo_junctionA)
                    edges_to_process_pseudojunctions.push_back(edge);
            {
                int edges_left = processEdges(edges_to_process_pseudojunctions, locatedPoints, index, addedPoints);
                assert(edges_left == 0);
            }
        }

        // then process edges that connect junctions to old nodes
        std::vector<Edge> edges_to_process_old;
        for (auto& edge : edges_in_1)
            if (edge.valid() && edge.indexA_ < finishedPoints && locatedPoints[edge.indexB_].type == point::Type::junction)
                edges_to_process_old.push_back(edge);
        for (auto& edge : edges_in_2)
            if (edge.valid() && edge.indexA_ < finishedPoints && locatedPoints[edge.indexB_].type == point::Type::junction)
                edges_to_process_old.push_back(edge);
        {
            int edges_left = processEdges(edges_to_process_old, locatedPoints, index, addedPoints);
            assert(edges_left == 0);
        }

        // then process edges that connect new junctions to new junctions (strange case)
        std::vector<Edge> edges_to_process_junc;
        for (auto& edge : edges_in_1)
            if (edge.valid() && edge.indexA_ >= finishedPoints &&
                locatedPoints[edge.indexA_].type == point::Type::junction &&
                locatedPoints[edge.indexB_].type == point::Type::junction)
                edges_to_process_junc.push_back(edge);
        for (auto& edge : edges_in_2)
            if (edge.valid() && edge.indexA_ >= finishedPoints &&
                locatedPoints[edge.indexA_].type == point::Type::junction &&
                locatedPoints[edge.indexB_].type == point::Type::junction)
                edges_to_process_junc.push_back(edge);
        {
            int edges_left = processEdges(edges_to_process_junc, locatedPoints, index, addedPoints);
            assert(edges_left == 0);
        }


        // then process all other edges
        while (true) {
            int edges_1_left = processEdges(edges_in_1, locatedPoints, index, addedPoints);
            int edges_2_left = processEdges(edges_in_2, locatedPoints, index, addedPoints);
            int sum = edges_1_left + edges_2_left;
            if (sum == 0)
                break;
        }

        // at this moment, all Points should have been added to the graph
        for (auto& point : locatedPoints)
            assert(point.number >= 0);

        // draw graph into visualisation to check it
        //data1.source[index].copyTo(data1.visualisation[index]);
        //data2.source[index].copyTo(data2.visualisation[index]);
        data1.drawGraph(*graph.root, index);
        data2.drawGraph(*graph.root, index);


        if (index != data1.size-1) {
        // when all are matched, correlate those matches to next endpoints layer / skeleton
        // junctions are expected to stay, however if a path appears again its now a (final) endpoint
        // for a match: delete the respective pixel in the endpoints layer
        // paint all known matches in the buffer layer of the next time step (at correct location)
        cv::bitwise_not(data1.endpoints[index+1], data1.buffer[index+1]);
        cv::bitwise_not(data2.endpoints[index+1], data2.buffer[index+1]);
        for (int i = 0; i < locatedPoints.size(); i++) {
            auto& point = locatedPoints[i];


            if ((point.status == point::Status::fresh || point.status == point::Status::frontier) && (point.type == point::Type::endpoint)) {
                // if it matches to another endpoint, it will stay, so change the type
                // ATTENTION: way too fancy trick with short circuit evaluation
                bool check1 = point.origin_1 && (pointCloseTo(data1.endpoints[index + 1], data1.visualisation[index + 1], 255, point.pos_1) != -1);
                bool check2 = point.origin_2 && (pointCloseTo(data2.endpoints[index + 1], data2.visualisation[index + 1], 255, point.pos_2) != -1);
                if (check1 || check2) {
                    // found another endpoint
                    point.type = point::Type::endpoint;
                    point.status = point::Status::old;
                }
                else {
                    point.type = point::Type::path;
                    point.status = point::Status::frontier;
                    // match the path to the skeleton (be sure to match!)
                    assert (pointCloseTo(data1.skeleton[index + 1], data1.visualisation[index + 1], 255, point.pos_1) >
                            -1);
                    assert (pointCloseTo(data2.skeleton[index + 1], data2.visualisation[index + 1], 255, point.pos_2) >
                            -1);
                    }
            }
            // if we have pseudo junctions that are not yet old, they are frontiers. we manually set them old if they match
            else if ((point.type == point::Type::pseudo_junctionA || point.type == point::Type::pseudo_junctionB) &&
                point.status != point::Status::old) {
                    point.status = point::Status::frontier;

            }
            else // only fresh endpoints can become frontiers, the rest just becomes old
                point.status = point::Status::old;

            if (point.type == point::Type::endpoint) {
                // a discovered and verified endpoint MUST stay an endpoint!
                if (point.origin_1) assert(pointCloseTo(data1.endpoints[index + 1], data1.visualisation[index + 1], 255, point.pos_1) != -1);
                if (point.origin_2) assert(pointCloseTo(data2.endpoints[index + 1], data2.visualisation[index + 1], 255, point.pos_2) != -1);
            }

            else if (point.type == point::Type::junction) {
                // a junction MUST still be a junction! if we are unable to match, something has seriously gone wrong.
                if (point.origin_1) assert(pointCloseTo(data1.endpoints[index + 1], data1.visualisation[index + 1], 128, point.pos_1) != -1);
                if (point.origin_2) assert(pointCloseTo(data2.endpoints[index + 1], data2.visualisation[index + 1], 128, point.pos_2) != -1);
            }

            else if (point.type == point::Type::pseudo_junctionA) {
                // a junction MUST still be a junction! if we are unable to match, something has seriously gone wrong.
                assert (pointCloseTo(data1.endpoints[index+1], data1.visualisation[index+1], 128, point.pos_1, 30) != -1);
                assert (pointCloseTo(data2.skeleton[index+1], data2.visualisation[index+1], 255, point.pos_2) > -1);
            }

            else if (point.type == point::Type::pseudo_junctionB) {
                // a junction MUST still be a junction! if we are unable to match, something has seriously gone wrong.
                assert (pointCloseTo(data2.endpoints[index+1], data2.visualisation[index+1], 128, point.pos_2, 30) != -1);
                assert (pointCloseTo(data1.skeleton[index+1], data1.visualisation[index+1], 255, point.pos_1) > -1);
            }

            else if (point.type == point::Type::path) {
                // match the path to the skeleton (be sure to match!)
                assert (pointCloseTo(data1.skeleton[index+1], data1.visualisation[index+1], 255, point.pos_1) > -1);
                assert (pointCloseTo(data2.skeleton[index+1], data2.visualisation[index+1], 255, point.pos_2) > -1);
            }




            // update node position
            Camera::intersect(data1.cam, point.pos_1, data2.cam, point.pos_2, point.node->position);
            // remove on endpoints layer
            data1.endpoints[index+1].at<uchar>(point.pos_1.y(), point.pos_1.x()) = 0;
            data2.endpoints[index+1].at<uchar>(point.pos_2.y(), point.pos_2.x()) = 0;
            // index on buffer
            data1.buffer[index+1].at<uchar>(point.pos_1.y(), point.pos_1.x()) = i;
            data2.buffer[index+1].at<uchar>(point.pos_2.y(), point.pos_2.x()) = i;
            // remove from next skeleton
            if (point.origin_1)
                data1.skeleton[index+1].at<uchar>(point.pos_1.y(), point.pos_1.x()) = 0;
            if (point.origin_2)
                data2.skeleton[index+1].at<uchar>(point.pos_2.y(), point.pos_2.x()) = 0;
        }
        }
        //if (index == data1.size-1)

        std::string egal;
        data1.write_to_file(egal );
        //debugWaitShow();

        // declare this part as finished!
        finishedPoints = locatedPoints.size();

        former_edges_in_1 = edges_in_1;
        former_edges_in_2 = edges_in_2;
    }

    debugWaitShow();
}

// finds the closest point with specific value close to position. modifies position and returns how many pixels difference we had
int pointCloseTo(const cv::Mat& img, cv::Mat& vis, uchar value, Vector2d& position, int range) {

    for (int k = 0; k<range; k++) {
        for (int i = -k; i <= k; i++) {
            if (i == k || i == -k) {
                for (int j = -k; j <= k; j++) {
                    Vector2d location = position + Vector2d(i, j);

                    if (img.at<uchar>(location.y(), location.x()) == value) {
                        position = location;
                        return std::max(std::abs(i),std::abs(j));
                    }
                    vis.at<uchar>(location.y(), location.x()) = 66;
                }
            }
            else {
                Vector2d location = position + Vector2d(i, -k);
                if (img.at<uchar>(location.y(), location.x()) == value) {
                    position = location;
                    return std::max(std::abs(i),std::abs(k));
                }
                vis.at<uchar>(location.y(), location.x()) = 66;
                location = position + Vector2d(i, +k);
                if (img.at<uchar>(location.y(), location.x()) == value) {
                    position = location;
                    return std::max(std::abs(i),std::abs(k));
                }
                vis.at<uchar>(location.y(), location.x()) = 66;
            }
        }
    }

    // signal failure
    return -1;
}

// tries to find best match, returns distance
double bestMatch(cv::Mat& img1, Camera& cam1, Vector2d& pix1, cv::Mat& img2, Camera& cam2, Vector2d& pix2, Vector3d& position) {

    // find point in skeleton that is == 255 for img1 and 2
    Vector2d candidate1;
    bool candidate1found = false;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            if (i != 0 || j != 0) {
                Vector2d trialPoint = pix1 + Vector2d(i, j);
                if (img1.at<uchar>(trialPoint.y(), trialPoint.x()) == 0) {
                    candidate1 = trialPoint;
                    assert (!candidate1found);
                    candidate1found = true;
                }
            }

    Vector2d candidate2;
    bool candidate2found = false;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            if (i != 0 || j != 0) {
                Vector2d trialPoint = pix2 + Vector2d(i, j);
                if (img2.at<uchar>(trialPoint.y(), trialPoint.x()) == 0) {
                    candidate2 = trialPoint;
                    assert (!candidate2found);
                    candidate2found = true;
                }
            }

    assert(candidate1found && candidate2found);

    // to which do these candidates fit best?
    double best1, best2;
    Vector2d pix2_new, pix1_new;
    Vector3d pos1, pos2;
    correlate(img2, cam1, cam2, candidate1, pix2, pix2_new, pos1, best1, 1, 2);
    correlate(img1, cam2, cam1, candidate2, pix1, pix1_new, pos2, best2, 1, 2);

    std::cout << "Match1: " << best1 << " Match2: " << best2 << std::endl;

    if (best1 < best2) {
        // never go back if we left!
        img1.at<uchar>(pix1.y(), pix1.x()) = 2;
        pix1 = candidate1;
        pix2 = pix2_new;
        position = pos1;
        return best1;
    }
    img2.at<uchar>(pix2.y(), pix2.x()) = 2;
    pix2 = candidate2;
    pix1 = pix1_new;
    position = pos2;
    return best2;
}

bool trace(imageData& data1, imageData& data2, int index, specialPoint& point,
                  std::vector<specialPoint>& specialPoints) {

    assert (point.found_1 && point.found_2 && point.node);
    Vector2d locA = point.pos_1;
    Vector2d locB = point.pos_2;

nextPixel:
    // mark pixel in skeleton as processed
    data1.skeleton[index].at<uchar>(locA.y(), locA.x()) = 1;
    data2.skeleton[index].at<uchar>(locB.y(), locB.x()) = 1;

    Vector3d position;
    // directly get new positions where we matched
    double distance = bestMatch(data1.skeleton[index], data1.cam, locA,
                                data2.skeleton[index], data2.cam, locB,
                                position);

    // check if pixel is special:
    int otherIndex = -1;

    if (data1.buffer[index].at<uchar>(locA.y(), locA.x()) != 255)
        otherIndex = data1.buffer[index].at<uchar>(locA.y(), locA.x());
    else if (data2.buffer[index].at<uchar>(locB.y(), locB.x()) != 255)
        otherIndex = data2.buffer[index].at<uchar>(locB.y(), locB.x());
    // was not special, continue trace (make sure we dont think our pixel is special)
    if (otherIndex == -1 || otherIndex == point.index)
        goto nextPixel;
    else {
        // it was special!!!
        specialPoint& other = specialPoints[otherIndex];

        if (other.node)
            point.node->graph.connectNodes(point.node, other.node);
        else
            other.node = point.node->addNode(position);

        if (!other.found_1) {
            other.found_1 = true;
            other.pos_1 = locA;
        }
        if (!other.found_2) {
            other.found_2 = true;
            other.pos_2 = locB;
        }
        data1.skeleton[index].at<uchar>(locA.y(), locA.x()) = 1;
        data2.skeleton[index].at<uchar>(locB.y(), locB.x()) = 1;
        return true;
    }
}

int processEdges(const std::vector<Edge>& edges, std::vector<specialPoint>& points, int index, int& added_edges){

    int open_edges = 0;

    // in an ideal world, all edges would now connect two locatedPoints
    for (auto& edge : edges)
        // both sides connected of the edge
        if(edge.indexA_ != -1 && edge.indexB_ != -1) {
        // we only need to do something if one of the nodes is not yet in the graph
        specialPoint& A = points[edge.indexA_];
        specialPoint& B = points[edge.indexB_];
        if (A.node != nullptr && B.node != nullptr){}
            // do nothing
        else if (A.node != nullptr) {
            assert(A.found_1 && A.found_2 && A.node && B.node == nullptr);

            // we know the point B is not yet added to the graph
            // if it is already found on both cameras, we have good location and can add to the graph
            if (B.found_1 && B.found_2) {
                assert(!B.node && A.node);
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(B.pos_1).normalized(),
                                                    data2.cam.origin, data2.cam.ray(B.pos_2).normalized(), position);
                assert (distance < 2);

                B.node = A.node->addNode(position);
                B.number = added_edges;
                added_edges++;
            }
            // point is only on one side yet
            else {
                if (!B.found_1) {
                    assert(B.found_2);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.cam.ray(B.pos_2));
                    data1.renderLine(line, index);
                    B.pos_1 = double_locate(data1.new_skeleton[index], data1.skeleton[index], data1.components[index], line, A.pos_1,
                                            data2.cam, B.pos_2, data1.cam);
                    if (B.pos_1 == Vector2d(-1, -1))
                        debugWaitShow();
                    B.found_1 = true;
                }
                if (!B.found_2) {
                    assert(B.found_1);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.cam.ray(B.pos_1));
                    data2.renderLine(line, index);
                    B.pos_2 = double_locate(data2.new_skeleton[index], data2.skeleton[index], data2.components[index], line, A.pos_2,
                                            data1.cam, B.pos_1, data2.cam);
                    if (B.pos_2 == Vector2d(-1, -1))
                        debugWaitShow();
                    B.found_2 = true;
                }
                assert(!B.node);
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(B.pos_1).normalized(),
                                                    data2.cam.origin, data2.cam.ray(B.pos_2).normalized(), position);
                assert (distance < 2);
                B.node = A.node->addNode(position);
                B.number = added_edges;
                added_edges++;
            }
        }
        else if (B.node) {
            assert(B.found_1 && B.found_2 && B.node && A.number == -1);

            // we know the point B is not yet added to the graph
            // if it is already found on both cameras, we have good location and can add to the graph
            if (A.found_1 && A.found_2) {
                assert(!A.node && B.node);
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(A.pos_1).normalized(),
                                                    data2.cam.origin, data2.cam.ray(A.pos_2).normalized(), position);
                assert (distance < 2);

                A.node = B.node->addNode(position);
                A.number = added_edges;
                added_edges++;
            }

                // point is only on one side yet
            else {
                if (!A.found_1) {
                    assert(A.found_2);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.cam.ray(A.pos_2));
                    data1.renderLine(line, index);
                    A.pos_1 = double_locate(data1.new_skeleton[index], data1.skeleton[index], data1.components[index], line, B.pos_1,
                                            data2.cam, A.pos_2, data1.cam);
                    if (A.pos_1 == Vector2d(-1, -1))
                        debugWaitShow();
                    A.found_1 = true;
                }
                if (!A.found_2) {
                    assert(A.found_1);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.cam.ray(A.pos_1));
                    data2.renderLine(line, index);
                    A.pos_2 = double_locate(data2.new_skeleton[index], data2.skeleton[index], data2.components[index], line, B.pos_2,
                                            data1.cam, A.pos_1, data2.cam);
                    if (A.pos_2 == Vector2d(-1, -1))
                        debugWaitShow();
                    A.found_2 = true;
                }
                assert(!B.node);
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(A.pos_1).normalized(),
                                                    data2.cam.origin, data2.cam.ray(A.pos_2).normalized(), position);
                assert (distance < 2);
                A.node = B.node->addNode(position);
                A.number = added_edges;
                added_edges++;
            }
        }
        else
            // both special points associated with this edge were not yet added to the graph
            open_edges ++;

    }
    return open_edges;
}

void displayVisual( int, void* )
{
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
}

void changeVisual( int pos, void* )
{

    std::string text = "Image source: ";
    from f;
    if (pos == 0) {
        f = from::source;
        text.append("source");
    }
    if (pos == 1) {
        f = from::skeleton;
        text.append("skeleton");
    }
    if (pos == 2) {
        f = from::new_skeleton;
        text.append("new_skeleton");
    }
    if (pos == 3) {
        f = from::components;
        text.append("components");
    }
    if (pos == 4) {
        f = from::endpoints;
        text.append("endpoints");
    }
    if (pos == 5) {

        f = from::buffer;
        text.append("buffer");
    }
    if (pos == 6) {
        f = from::visualisation;
        text.append("visualisation");
    }

    data1.resetVisual(f);
    data2.resetVisual(f);
    cv::displayStatusBar("Control", text);
    displayVisual(0,0);
}

