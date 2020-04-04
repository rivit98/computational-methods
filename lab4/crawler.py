from wikipedia import Wikipedia
import argparse

''' A script used to randomly collect Wikipedia articles '''

''' Parse command line arguments '''

parser = argparse.ArgumentParser()

parser.add_argument(
	"how_many_pages",
	type=int,
	help="crawling articles limit"
)
parser.add_argument(
	"subdomain",
	type=str,
	help="crawling subdomain"
)

args = parser.parse_args()

''' Start crawling '''

wiki = Wikipedia(args.subdomain)
wiki.crawl(args.how_many_pages)
