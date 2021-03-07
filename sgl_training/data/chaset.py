import os
from data import dbdata_chase as dbdata

class CHASET(dbdata.DBData):
    def __init__(self, args, name='CHASET', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.train = train
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(CHASET, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr  = super(CHASET, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        return names_hr

    def _set_filesystem(self, dir_data):
        super(CHASET, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
