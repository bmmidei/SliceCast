class Document(object):
    
    def __init__(self, rawText):
        # Initialize with raw text
        self.raw = rawText

    def getSegments(self, delim):
        # Split segments on the segment delimiter
        # TODO need to remove the section number and title

        # TODO Each item begins with a section delimiter, therefore the
        # firs segment is an empty string. Need to fix this
        rawSegs = self.raw.split(sep=delim)
        return rawSegs



class Segment(object):

    def __init__(self, rawText):
        self.raw = rawText
