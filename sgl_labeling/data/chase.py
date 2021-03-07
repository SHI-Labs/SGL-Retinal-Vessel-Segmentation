import os
from data import dbdata_chase as dbdata

class CHASE(dbdata.DBData):
    def __init__(self, args, name='CHASE', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.train = train
        if train:
            data_range = data_range[0:-1][0]  #all the split
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[-1]  #the last one
        self.tl = list(map(lambda x: int(x), data_range))
        print(self.tl)
        super(CHASE, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        super_names_hr  = super(CHASE, self)._scan()
        names_hr = []
        for i in range(int(len(self.tl)/2)):
            names_hr.extend(super_names_hr[self.tl[2 * i] - 1 : self.tl[2 * i + 1]])
        print(names_hr)
        return names_hr

    def _set_filesystem(self, dir_data):
        super(CHASE, self)._set_filesystem(dir_data)
