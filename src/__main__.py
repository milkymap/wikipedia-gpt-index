
import click 

from dotenv import load_dotenv

from src.log import logger 
from src.cli import handle_video

@click.group()
@click.pass_context
@click.option('--openai_api_key', envvar='OPENAI_API_KEY', required=True)
@click.option('--telegram_token', envvar='TELEGRAM_TOKEN', required=True)
def handler(ctx:click.core.Context, openai_api_key:str, telegram_token:str):
    ctx.ensure_object(dict)
    subcommand = ctx.invoked_subcommand
    logger.info(f'{subcommand} was called')
    ctx.obj['keys'] = (openai_api_key, telegram_token)

handler.add_command(cmd=handle_video)  

if __name__ == '__main__':
    load_dotenv()
    handler()