import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir,name, title, makehtml=False, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.name=name
        self.makehtml=makehtml
        if makehtml:
            self.img_dir = os.path.join(self.web_dir, '{:s}'.format(name))
        else:
            self.img_dir = os.path.join(self.web_dir, 'images_{:s}'.format(name))
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=256):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):

                    # print('im', im,'txt', txt, 'link', link)
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            if self.makehtml:
                                with a(href=os.path.join('{:s}'.format(self.name), link)):
                                    img(style="width:%dpx" % (width), src=os.path.join('{:s}'.format(self.name), im),)
                                br()
                                p(txt)
                            else:
                                with a(href=os.path.join('images_{:s}'.format(self.name), link)):
                                    img(style="width:%dpx" % (width), src=os.path.join('images_{:s}'.format(self.name), im),)
                                br()
                                p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()
