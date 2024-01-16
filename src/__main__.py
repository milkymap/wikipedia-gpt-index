
import click 

from dotenv import load_dotenv

from src.log import logger 
from src.cli import build_index_from_wikipedia

@click.group()
@click.pass_context
@click.option('--openai_api_key', envvar='OPENAI_API_KEY', required=True)
@click.option('--telegram_token', envvar='TELEGRAM_TOKEN', required=True)
@click.option('--cache_folder', envvar='TRANSFORMERS_CACHE', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--device', default='cpu')
def handler(ctx:click.core.Context, openai_api_key:str, telegram_token:str, cache_folder:str, device:str):
    ctx.ensure_object(dict)
    subcommand = ctx.invoked_subcommand
    logger.info(f'{subcommand} was called')

    ctx.obj['openai_api_key'] = openai_api_key
    ctx.obj['telegram_token'] = telegram_token

    ctx.obj['cache_folder'] = cache_folder
    ctx.obj['device'] = device

handler.add_command(cmd=build_index_from_wikipedia, name='wikipedia')  

if __name__ == '__main__':
    load_dotenv()
    handler()