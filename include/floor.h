#ifndef INTELLIGENTCABINET_FLOOR_H
#define INTELLIGENTCABINET_FLOOR_H

#include <map>
#include <iostream>

using namespace std;

class Floor {
public:
    Floor(int index);
    ~Floor();

private:
    map<string, int> m_contains;
    int m_index;
};


#endif //INTELLIGENTCABINET_FLOOR_H
