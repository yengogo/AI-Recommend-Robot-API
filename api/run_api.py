#!/usr/bin/env python
# coding: utf-8
# from model import tour_guide_generate
import uvicorn

if __name__ == '__main__':
    uvicorn.run('model.tour_guide_generate:app',
                host='0.0.0.0', port=10001, reload=True)
